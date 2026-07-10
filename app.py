import unicodedata

import chainlit as cl
from mcp import ClientSession

from meeting_agent import extraction, mcp_tools, notion_mapping
from meeting_agent.models import Task, TaskList
from meeting_agent.state import get_state, reset_state

WELCOME = """**Meeting Tasks Agent**

Paste a meeting transcript below (or attach a `.txt`/`.md` file) and I'll pull out the action items — owner, due date, and dependencies — for you to review.

When you're ready to create the tasks in Notion, connect a Notion MCP server via the 🔌 icon in the composer (either the official `npx -y @notionhq/notion-mcp-server`, or Notion's hosted MCP at `https://mcp.notion.com/mcp`). You can do this any time before confirming."""


def _render_task_list(tasks: list[Task]) -> str:
    if not tasks:
        return "_No action items found._"
    lines = []
    for i, t in enumerate(tasks, 1):
        meta = []
        if t.owner:
            meta.append(f"owner: {t.owner}")
        if t.due_date:
            meta.append(f"due: {t.due_date}")
        if t.dependencies:
            meta.append(f"depends on: {', '.join(t.dependencies)}")
        meta_str = f" _({'; '.join(meta)})_" if meta else ""
        lines.append(f"{i}. **{t.title}**{meta_str}\n   {t.description}")
    return "\n\n".join(lines)


async def _extract_from_message(message: cl.Message) -> str | None:
    for element in message.elements:
        if element.path:
            with open(element.path, "rb") as f:
                raw = f.read()
            return unicodedata.normalize("NFC", raw.decode("utf-8", errors="replace"))
    if message.content.strip():
        return unicodedata.normalize("NFC", message.content)
    return None


@cl.on_chat_start
async def on_chat_start():
    reset_state()
    await cl.Message(content=WELCOME).send()


@cl.on_mcp_connect
async def on_mcp_connect(connection, session: ClientSession):
    state = get_state()
    count = await mcp_tools.register_mcp_connection(state, connection.name, session)
    await cl.Message(content=f"Connected MCP server `{connection.name}` — {count} tools available.").send()


@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session: ClientSession):
    state = get_state()
    mcp_tools.unregister_mcp_connection(state, name)


async def _handle_collecting_transcript(message: cl.Message, state):
    transcript = await _extract_from_message(message)
    if not transcript:
        await cl.Message(content="Please paste some transcript text or attach a file.").send()
        return

    state.transcript = transcript
    thinking = cl.Message(content="Extracting tasks…")
    await thinking.send()
    task_list = await extraction.extract_tasks(transcript)
    state.tasks = task_list.tasks
    state.stage = "reviewing_tasks"

    thinking.content = (
        f"**Extracted {len(state.tasks)} task(s):**\n\n{_render_task_list(state.tasks)}\n\n"
        "Reply with any corrections (e.g. \"merge tasks 2 and 3\", \"John is the owner of task 1\"), "
        "or click **Looks good** below to proceed."
    )
    thinking.actions = [
        cl.Action(name="proceed_to_confirm", payload={}, label="✅ Looks good, proceed"),
    ]
    await thinking.update()


async def _handle_reviewing_tasks(message: cl.Message, state):
    feedback = message.content.strip()
    if not feedback:
        return
    thinking = cl.Message(content="Revising…")
    await thinking.send()
    task_list = await extraction.revise_tasks(TaskList(tasks=state.tasks), feedback)
    state.tasks = task_list.tasks
    thinking.content = f"**Revised task list ({len(state.tasks)} task(s)):**\n\n{_render_task_list(state.tasks)}"
    thinking.actions = [
        cl.Action(name="proceed_to_confirm", payload={}, label="✅ Looks good, proceed"),
    ]
    await thinking.update()


@cl.action_callback("proceed_to_confirm")
async def on_proceed_to_confirm(action: cl.Action):
    state = get_state()
    state.stage = "confirming"
    await _run_confirmation_flow()


async def _run_confirmation_flow():
    state = get_state()

    if not state.mcp_clients:
        await cl.Message(
            content="No Notion MCP server is connected yet. Connect one via the 🔌 icon in the composer, then send any message to continue."
        ).send()
        return

    discovery_prompt = (
        "You are helping a user pick which Notion database (data source) their extracted meeting tasks "
        "should be created in. Use the available tools to search/list data sources, then reply with a short "
        "plain-text summary of the best candidate data source, ending your reply with a line exactly of the form "
        "`DATA_SOURCE_ID: <id>` and `DATA_SOURCE_NAME: <name>`."
    )
    task_titles = ", ".join(t.title for t in state.tasks)
    user_prompt = f"The extracted tasks to create: {task_titles}. Find the most likely target Notion database."

    reply, _ = await mcp_tools.run_tool_calling_loop(
        state, discovery_prompt, [{"role": "user", "content": user_prompt}]
    )

    data_source_id = None
    data_source_name = None
    for line in reply.splitlines():
        if line.strip().startswith("DATA_SOURCE_ID:"):
            data_source_id = line.split(":", 1)[1].strip()
        if line.strip().startswith("DATA_SOURCE_NAME:"):
            data_source_name = line.split(":", 1)[1].strip()

    if not data_source_id:
        await cl.Message(
            content=f"Couldn't determine a target Notion database automatically.\n\n{reply}\n\nTell me the database name and try again."
        ).send()
        state.stage = "reviewing_tasks"
        return

    state.notion_data_source_id = data_source_id
    state.notion_data_source_name = data_source_name

    schema = await notion_mapping.fetch_data_source_schema(state, data_source_id)
    mapping = notion_mapping.compute_property_mapping(schema)
    state.property_mapping = mapping

    mapping_lines = [
        f"- Title → `{mapping['title']}`" if mapping["title"] else "- Title → (no title property found!)",
        f"- Owner → `{mapping['owner']}`" if mapping["owner"] else "- Owner → page body text (no matching property)",
        f"- Due date → `{mapping['due_date']}`" if mapping["due_date"] else "- Due date → page body text (no matching property)",
        "- Dependencies → page body text",
    ]

    res = await cl.AskActionMessage(
        content=(
            f"Ready to create **{len(state.tasks)}** task(s) in Notion database **{data_source_name}**.\n\n"
            "Field mapping:\n" + "\n".join(mapping_lines)
        ),
        actions=[
            cl.Action(name="confirm", payload={"decision": "confirm"}, label="✅ Create tasks"),
            cl.Action(name="edit", payload={"decision": "edit"}, label="✏️ Keep editing"),
            cl.Action(name="cancel", payload={"decision": "cancel"}, label="❌ Cancel"),
        ],
        timeout=300,
    ).send()

    decision = res.get("payload", {}).get("decision") if res else "cancel"

    if decision == "confirm":
        state.write_confirmed = True
        state.stage = "creating"
        await _create_tasks()
    elif decision == "edit":
        state.stage = "reviewing_tasks"
        await cl.Message(content="Okay, what would you like to change?").send()
    else:
        state.stage = "reviewing_tasks"
        await cl.Message(content="Cancelled. No tasks were created.").send()


async def _create_tasks():
    state = get_state()
    if not state.write_confirmed or not state.notion_data_source_id or not state.property_mapping:
        await cl.Message(content="Internal error: write was not properly confirmed.").send()
        return

    progress = cl.Message(content=f"Creating {len(state.tasks)} task(s) in Notion…")
    await progress.send()

    results = await notion_mapping.create_tasks_in_notion(
        state, state.notion_data_source_id, state.property_mapping, state.tasks
    )

    lines = []
    for r in results:
        icon = "✅" if r["ok"] else "❌"
        lines.append(f"{icon} {r['task']}" + ("" if r["ok"] else f" — {r['detail']}"))

    progress.content = "**Done:**\n\n" + "\n".join(lines)
    await progress.update()
    state.stage = "done"


@cl.on_message
async def on_message(message: cl.Message):
    state = get_state()

    if state.stage == "collecting_transcript":
        await _handle_collecting_transcript(message, state)
    elif state.stage == "reviewing_tasks":
        await _handle_reviewing_tasks(message, state)
    elif state.stage == "confirming":
        # user sent a message while we're mid-confirmation (e.g. just connected MCP) — retry
        await _run_confirmation_flow()
    elif state.stage in ("creating", "done"):
        await cl.Message(
            content="This session's tasks are already handled. Paste a new transcript to start over."
        ).send()
        reset_state()
        await _handle_collecting_transcript(message, get_state())
