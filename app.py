import unicodedata

import chainlit as cl
from mcp import ClientSession

from meeting_agent import extraction, mcp_tools, notion_mapping
from meeting_agent.langchain_llm import LLMResponseError
from meeting_agent.logging_config import get_logger, setup_logging
from meeting_agent.models import Task, TaskList
from meeting_agent.state import get_state, reset_state

setup_logging()
log = get_logger("app")

WELCOME = """**Meeting Tasks Agent**

Paste a meeting transcript below (or attach a `.txt`/`.md` file) and I'll pull out the action items — owner, due date, and dependencies — for you to review.

When you're ready to create the tasks in Notion, connect Notion's hosted MCP server via the 🔌 icon in the composer: add a `stdio` server with command `npx -y mcp-remote https://mcp.notion.com/mcp`. The first time, it'll open a browser for you to sign in to Notion (OAuth) — after that it reconnects automatically. You can do this any time before confirming."""


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
        if t.status:
            meta.append(f"status: {t.status}")
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
    log.info("chat_start: new session")
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
    try:
        task_list = await extraction.extract_tasks(transcript)
    except LLMResponseError as e:
        log.exception("extract_tasks failed")
        thinking.content = f"⚠️ Couldn't extract tasks: {e}"
        await thinking.update()
        return
    state.tasks = task_list.tasks
    state.stage = "reviewing_tasks"
    log.info("stage: collecting_transcript -> reviewing_tasks")

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
    try:
        task_list = await extraction.revise_tasks(TaskList(tasks=state.tasks), feedback)
    except LLMResponseError as e:
        log.exception("revise_tasks failed")
        thinking.content = f"⚠️ Couldn't revise tasks: {e}"
        await thinking.update()
        return
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
    log.info("stage: reviewing_tasks -> confirming")
    await _run_confirmation_flow()


async def _run_confirmation_flow():
    state = get_state()

    if not state.mcp_clients:
        await cl.Message(
            content="No Notion MCP server is connected yet. Click the 🔌 icon in the composer, add a `stdio` server with command `npx -y mcp-remote https://mcp.notion.com/mcp`, sign in with Notion in the browser window that opens, then send any message to continue."
        ).send()
        return

    task_titles = ", ".join(t.title for t in state.tasks)

    try:
        candidates_result = await notion_mapping.discover_data_sources(state, task_titles)
    except LLMResponseError as e:
        log.exception("discover_data_sources failed")
        await cl.Message(content=f"⚠️ Couldn't discover a Notion database: {e}").send()
        state.stage = "reviewing_tasks"
        return
    candidates = candidates_result.candidates

    if not candidates:
        await cl.Message(
            content="Couldn't find any Notion database automatically. "
            "Make sure the target database is shared with your Notion connection, then try again."
        ).send()
        state.stage = "reviewing_tasks"
        return

    candidate_lines = "\n".join(
        f"- **{c.name}**" + (f" — {c.url}" if c.url else "") for c in candidates
    )
    res = await cl.AskActionMessage(
        content=f"Found {len(candidates)} candidate Notion database(s):\n\n{candidate_lines}\n\nWhich one should tasks be created in?",
        actions=[
            cl.Action(name="pick_data_source", payload={"id": c.id, "name": c.name}, label=c.name)
            for c in candidates
        ]
        + [
            cl.Action(name="create_new_data_source", payload={}, label="➕ Create a new database"),
            cl.Action(name="cancel_pick", payload={}, label="❌ None of these / Cancel"),
        ],
        timeout=300,
    ).send()

    if not res or res.get("name") == "cancel_pick":
        await cl.Message(content="Cancelled. No tasks were created.").send()
        state.stage = "reviewing_tasks"
        return

    if res.get("name") == "create_new_data_source":
        name_msg = await cl.AskUserMessage(content="What should the new database be called?", timeout=300).send()
        if not name_msg or not name_msg.get("output", "").strip():
            await cl.Message(content="Cancelled. No tasks were created.").send()
            state.stage = "reviewing_tasks"
            return
        new_db_name = name_msg["output"].strip()

        parent_msg = await cl.AskUserMessage(
            content="Which Notion page should this new database be created in? (name or URL)", timeout=300
        ).send()
        if not parent_msg or not parent_msg.get("output", "").strip():
            await cl.Message(content="Cancelled. No tasks were created.").send()
            state.stage = "reviewing_tasks"
            return
        parent_page_name = parent_msg["output"].strip()

        creating_msg = cl.Message(content=f"Creating database **{new_db_name}** inside **{parent_page_name}**…")
        await creating_msg.send()
        try:
            created = await notion_mapping.create_data_source(state, new_db_name, parent_page_name)
        except LLMResponseError as e:
            log.exception("create_data_source failed")
            creating_msg.content = f"⚠️ Couldn't create the database: {e}"
            await creating_msg.update()
            state.stage = "reviewing_tasks"
            return
        data_source_id = created.id
        data_source_name = created.name
        creating_msg.content = f"Created database **{data_source_name}**" + (
            f" — {created.url}" if created.url else ""
        )
        await creating_msg.update()
    else:
        data_source_id = res["payload"]["id"]
        data_source_name = res["payload"]["name"]

    log.info("confirmation_flow: using data_source_id=%s name=%s", data_source_id, data_source_name)
    state.notion_data_source_id = data_source_id
    state.notion_data_source_name = data_source_name

    try:
        mapping = await notion_mapping.resolve_property_mapping(state, data_source_id)
    except Exception as e:
        log.exception("confirmation_flow: failed to resolve Notion property mapping")
        await cl.Message(content=f"⚠️ Couldn't read the Notion database schema: {e}").send()
        state.stage = "reviewing_tasks"
        return
    state.property_mapping = mapping
    log.info("confirmation_flow: property_mapping=%s", mapping.model_dump())

    try:
        state.existing_page_ids = await notion_mapping.find_existing_tasks(state, data_source_id, mapping, state.tasks)
    except LLMResponseError as e:
        log.exception("find_existing_tasks failed")
        await cl.Message(content=f"⚠️ Couldn't check for existing tasks, proceeding as if none exist: {e}").send()
        state.existing_page_ids = {}

    mapping_lines = [
        f"- Title → `{mapping.title_property}`" if mapping.title_property else "- Title → (no title property found!)",
        f"- Owner → `{mapping.owner_property}`" if mapping.owner_property else "- Owner → page body text (no matching property)",
        f"- Due date → `{mapping.due_date_property}`" if mapping.due_date_property else "- Due date → page body text (no matching property)",
        f"- Status → `{mapping.status_property}`" if mapping.status_property else "- Status → page body text (no matching property)",
        "- Dependencies → page body text",
    ]

    new_count = len(state.tasks) - len(state.existing_page_ids)
    update_count = len(state.existing_page_ids)
    summary_line = f"**{new_count}** new task(s) to create"
    if update_count:
        summary_line += f", **{update_count}** already exist and will just have their progress updated"

    res = await cl.AskActionMessage(
        content=(
            f"{summary_line} in Notion database **{data_source_name}**.\n\n"
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
    log.info("confirmation_flow: user decision=%s", decision)

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
        log.error(
            "create_tasks: internal error, write_confirmed=%s data_source_id=%s property_mapping=%s",
            state.write_confirmed,
            state.notion_data_source_id,
            state.property_mapping,
        )
        await cl.Message(content="Internal error: write was not properly confirmed.").send()
        return

    progress = cl.Message(content=f"Creating/updating {len(state.tasks)} task(s) in Notion…")
    await progress.send()

    results = await notion_mapping.create_tasks_in_notion(
        state, state.notion_data_source_id, state.property_mapping, state.tasks, state.existing_page_ids
    )

    action_labels = {"created": "created", "updated": "progress updated", "unchanged": "already up to date"}
    lines = []
    for r in results:
        icon = "✅" if r.ok else "❌"
        label = action_labels.get(r.action, r.action)
        lines.append(f"{icon} {r.task_title} ({label})" + ("" if r.ok else f" — {r.detail}"))
    log.info("create_tasks: done, %d/%d succeeded", sum(r.ok for r in results), len(results))

    progress.content = "**Done:**\n\n" + "\n".join(lines)
    await progress.update()
    state.stage = "done"


@cl.on_message
async def on_message(message: cl.Message):
    state = get_state()
    log.debug("on_message: stage=%s content_len=%d", state.stage, len(message.content or ""))

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
