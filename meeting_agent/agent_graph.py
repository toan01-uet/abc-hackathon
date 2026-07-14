import json
import unicodedata
from typing import List, TypedDict, Literal, cast

import chainlit as cl
from uuid import uuid4
import asyncio
from mcp import ClientSession

from langgraph.graph import StateGraph, END
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from meeting_agent import extraction, mcp_tools, notion_mapping
from meeting_agent.langchain_llm import FptStructuredRunnable, LLMResponseError
from meeting_agent.logging_config import get_logger
from meeting_agent.state import get_state, reset_state
from meeting_agent.models import Task, TaskList
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

setup_needed = False
log = get_logger("agent_graph")


# --- BƯỚC 1: ĐỊNH NGHĨA STATE VÀ CÁC ĐỐI TƯỢNG DỮ LIỆU (NỘI BỘ GRAPH) ---

# Định nghĩa cấu trúc cho một Task để dễ dàng quản lý bên trong graph (khác với Pydantic Task)
class GraphTask(TypedDict):
    assignee: str | None
    action_item: str
    due_date: str | None


class GraphAgentState(TypedDict, total=False):
    # Accumulated conversation history — full list passed on each invocation (no reducer, replaces in checkpoint)
    messages: list | None
    transcript: str | None
    tasks: List[GraphTask] | None
    notion_page_ids: List[str] | None
    error_message: str | None
    # UI / human-in-loop helpers
    awaiting_input: bool | None
    prompt: str | None
    last_user_input: str | None
    # HITL button schema: {buttons: [{label, value, style}], allow_freetext: bool, placeholder: str}
    # Produced by HITL nodes so drive_graph knows which buttons to render.
    hitl_interaction: dict | None
    # instructions for the external driver (side-effect calls)
    to_call: str | None
    # discover/create candidates
    candidates: List[dict] | None
    chosen_candidate_id: str | None
    # review state
    reviewed: bool | None
    revise_feedback: str | None
    # resolved mapping (as plain dict)
    mapping: dict | None
    existing_page_ids: dict | None
    # final decision/result
    decision: str | None
    results: List[dict] | None
    # optional explicit override route produced by an LLM router
    route_override: str | None


class RouteDecision(BaseModel):
    route: str | None = None
    prompt: str | None = None
    awaiting_input: bool | None = None
    state_updates: dict | None = None
    reply: str | None = None


async def ask_transcript(state: GraphAgentState) -> dict:
    # If we already have a transcript, nothing to do
    if state.get("transcript"):
        return {}

    # Build a small agent to craft a friendly transcript request for the user
    try:
        tools = mcp_tools.get_tools_by_name(get_state(), getattr(notion_mapping, "_READ_ONLY_DISCOVERY_TOOLS", set()))
    except Exception:
        tools = mcp_tools.get_all_tools(get_state())

    from meeting_agent.agent import build_notion_agent, run_agent

    agent = build_notion_agent(tools)

    user_msg = (
        "You are a friendly assistant asking the human to paste or attach the meeting transcript. "
        "Keep the prompt short and clear, and ask for any relevant context such as meeting date or participants."
    )

    try:
        final_text, _ = await run_agent(agent, "You are a prompt-crafting assistant.", [HumanMessage(content=user_msg)])
    except Exception:
        log.exception("ask_transcript: agent run failed")
        # Fallback to a minimal prompt
        final_text = "Please paste the meeting transcript (or attach a .txt/.md file)."

    return {"prompt": final_text, "awaiting_input": True}


def extract_tasks_node(state: GraphAgentState) -> dict:
    # Signal the driver to run the extraction step
    if not state.get("transcript"):
        return {"prompt": "No transcript available; please paste one.", "awaiting_input": True}
    return {"to_call": "extract_tasks"}


def ask_review(state: GraphAgentState) -> dict:
    tasks = state.get("tasks") or []
    if not tasks:
        return {"to_call": "extract_tasks"}
    lines = [f"{i+1}. **{t.get('action_item')}** (owner: {t.get('assignee') or 'unassigned'})" for i, t in enumerate(tasks)]
    prompt = (
        f"I extracted **{len(tasks)} task(s)**:\n\n" + "\n".join(lines) +
        "\n\nClick **Looks good** to proceed to Notion, or type a correction below."
    )
    hitl_interaction = {
        "buttons": [
            {"label": "✅ Looks good, proceed to Notion", "value": "looks good", "style": "primary"},
        ],
        "allow_freetext": True,
        "placeholder": "Type a correction (e.g. 'task 2 owner is Alice', 'merge 1 and 3')…",
    }
    return {"prompt": prompt, "awaiting_input": True, "hitl_interaction": hitl_interaction}


def process_review(state: GraphAgentState) -> dict:
    last = (state.get("last_user_input") or "").strip()
    if not last:
        return {"prompt": "Please provide feedback or 'looks good' to proceed.", "awaiting_input": True}
    _approval_phrases = ("look", "ok", "okay", "good", "proceed", "continue", "yes", "done", "approve")
    if any(p in last.lower() for p in _approval_phrases):
        # User approved the task list — advance to Notion integration stage
        return {"reviewed": True, "last_user_input": None}
    # User wants revisions — pass feedback to the driver then clear input so routing is clean
    return {"to_call": "revise_tasks", "revise_feedback": last, "last_user_input": None}


def discover_candidates_node(state: GraphAgentState) -> dict:
    return {"to_call": "discover_data_sources"}


def ask_pick_candidate(state: GraphAgentState) -> dict:
    candidates = state.get("candidates") or []
    if not candidates:
        return {"to_call": "discover_data_sources"}
    lines = [f"- **{c.get('name')}**" + (f" — {c.get('url')}" if c.get('url') else "") for c in candidates]
    prompt = "Found the following Notion databases:\n\n" + "\n".join(lines) + "\n\nSelect a database, create a new one, or cancel."
    db_buttons = [
        {"label": c.get("name") or c.get("id"), "value": c.get("name") or c.get("id"), "style": "secondary"}
        for c in candidates if c.get("name") or c.get("id")
    ]
    db_buttons += [
        {"label": "➕ Create a new database", "value": "__create_new__", "style": "secondary"},
        {"label": "❌ Cancel", "value": "cancel", "style": "danger"},
    ]
    hitl_interaction = {"buttons": db_buttons, "allow_freetext": False, "placeholder": ""}
    return {"prompt": prompt, "awaiting_input": True, "hitl_interaction": hitl_interaction}


def process_pick_candidate(state: GraphAgentState) -> dict:
    last = (state.get("last_user_input") or "").strip()
    if not last:
        return {"prompt": "Please pick a candidate by name or type 'create new' or 'cancel'.", "awaiting_input": True}
    if "create" in last.lower():
        return {"to_call": "create_data_source", "last_user_input": None}
    if "cancel" in last.lower():
        return {"error_message": "Cancelled by user", "last_user_input": None}
    # match by name or id
    for c in state.get("candidates") or []:
        name = c.get("name", "") or ""
        id_ = c.get("id")
        if name.lower() in last.lower() or (id_ and str(id_) in last):
            return {"chosen_candidate_id": id_, "last_user_input": None}
    return {"prompt": "Could not match your choice; reply with the exact database name or id.", "awaiting_input": True}


def resolve_mapping_node(state: GraphAgentState) -> dict:
    if not state.get("chosen_candidate_id"):
        return {"to_call": "discover_data_sources"}
    return {"to_call": "resolve_property_mapping"}


def confirm_create_node(state: GraphAgentState) -> dict:
    mapping = state.get("mapping") or {}
    title_prop = mapping.get("title_property") if mapping else None
    owner_prop = mapping.get("owner_property") if mapping else None
    due_prop   = mapping.get("due_date_property") if mapping else None
    status_prop = mapping.get("status_property") if mapping else None
    mapping_lines = [
        f"- Title → `{title_prop}`" if title_prop else "- Title → *(no title property found)*",
        f"- Owner → `{owner_prop}`" if owner_prop else "- Owner → *(page body)*",
        f"- Due date → `{due_prop}`" if due_prop else "- Due date → *(page body)*",
        f"- Status → `{status_prop}`" if status_prop else "- Status → *(page body)*",
    ]
    new_count = len(state.get("tasks") or [])
    prompt = (
        f"**{new_count} task(s)** will be created in the selected Notion database.\n\n"
        "**Field mapping:**\n" + "\n".join(mapping_lines)
        + "\n\nConfirm to create the tasks in Notion."
    )
    hitl_interaction = {
        "buttons": [
            {"label": "✅ Create tasks in Notion", "value": "confirm", "style": "primary"},
            {"label": "✏️ Edit tasks", "value": "edit", "style": "secondary"},
            {"label": "❌ Cancel", "value": "cancel", "style": "danger"},
        ],
        "allow_freetext": False,
        "placeholder": "",
    }
    return {"prompt": prompt, "awaiting_input": True, "hitl_interaction": hitl_interaction}


def process_confirm(state: GraphAgentState) -> dict:
    last = (state.get("last_user_input") or "").strip()
    if not last:
        return {"prompt": "Please respond 'confirm', 'edit', or 'cancel'.", "awaiting_input": True}
    if "confirm" in last.lower() or "yes" in last.lower() or "ok" in last.lower():
        return {"decision": "confirm", "to_call": "create_tasks_in_notion", "last_user_input": None}
    if "edit" in last.lower() or "change" in last.lower() or "revise" in last.lower():
        return {"decision": "edit", "last_user_input": None}
    return {"decision": "cancel", "last_user_input": None}


def create_tasks_node(state: GraphAgentState) -> dict:
    return {"to_call": "create_tasks_in_notion"}


async def decide_route_node(state: GraphAgentState) -> dict:
    """Tool-less router node: use a structured LLM call to pick the next route.

    This node intentionally does NOT call any MCP tools — it only asks the
    chat model to return a JSON `RouteDecision` and maps that into the graph
    output. To avoid self-referential loops, the node will never return the
    `decide_route` route; instead it falls back to a safe route.
    """
    # If a previous run already set an explicit override, consume it
    if state.get("route_override"):
        return {"route": state.get("route_override"), "state_updates": {"route_override": None}}

    # Summarize state for the router LLM
    tasks = state.get("tasks") or []
    tasks_preview = "\n".join([f"- {t.get('action_item')} (owner: {t.get('assignee')})" for t in tasks]) or "(no tasks)"
    # Build conversation history context from accumulated messages
    history_msgs = state.get("messages") or []
    if history_msgs:
        history_lines = []
        for _m in history_msgs[-8:]:
            _role = "User" if isinstance(_m, HumanMessage) else "Assistant"
            history_lines.append(f"{_role}: {str(_m.content)[:300]}")
        chat_context = "\n".join(history_lines)
    else:
        chat_context = "(no prior conversation)"
    user_msg = (
        "You are a concise routing assistant for a Meeting Tasks workflow.\n\n"
        f"Recent conversation:\n{chat_context}\n\n"
        f"State summary:\n- Transcript present: {bool(state.get('transcript'))}\n- Transcript length: {len(str(state.get('transcript') or ''))}\n- Last user input: {state.get('last_user_input')!r}\n- Tasks ({len(tasks)}):\n{tasks_preview}\n- Candidates: {state.get('candidates') or []}\n- Chosen candidate id: {state.get('chosen_candidate_id') or None}\n- Mapping present: {bool(state.get('mapping'))}\n\n"
        "Decide the best next route from: ask_transcript, extract_tasks, ask_review, process_review, discover_candidates, "
        "ask_pick_candidate, process_pick_candidate, resolve_mapping, confirm_create, process_confirm, create_tasks, __end__.\n\n"
        "Return ONLY a JSON object matching this schema: {\n  \"route\": string|null,\n  \"prompt\": string|null,\n  \"awaiting_input\": boolean|null,\n  \"state_updates\": object|null,\n  \"reply\": string|null\n}\n\n"
        "If you want to ask the human something, set 'awaiting_input' true and include 'prompt'."
    )

    runnable = FptStructuredRunnable(RouteDecision)
    llm_messages = [SystemMessage(content="You are a concise routing assistant."), HumanMessage(content=user_msg)]

    try:
        decision = await runnable.ainvoke(llm_messages)
        dd = decision.model_dump()
    except Exception:
        log.exception("decide_route_node: structured parse failed")
        # best-effort fallback based on state
        if not state.get("transcript"):
            return {"route": "ask_transcript"}
        if not state.get("tasks"):
            return {"route": "extract_tasks"}
        return {"route": "ask_review"}

    out: dict = {}
    # Apply state updates if any (driver will apply them too via state_updates path)
    if dd.get("state_updates"):
        out["state_updates"] = dd.get("state_updates")

    # If the LLM expects human input, surface prompt + awaiting flag and pick a safe route
    if dd.get("awaiting_input"):
        out["awaiting_input"] = True
        out["prompt"] = dd.get("prompt") or dd.get("reply")
        # ensure we don't return the routing node itself
        route = dd.get("route") or "ask_review"
        if route == "decide_route":
            route = "ask_review"
        out["route"] = route
        return out

    # If LLM returned a conversational reply only, surface it so driver can send
    if dd.get("reply") and not dd.get("route"):
        out["prompt"] = dd.get("reply")

    # Prefer an explicit route if provided. For routes that require an external
    # call (like 'extract_tasks'), surface a `to_call` so the driver performs
    # the action immediately instead of re-running the router.
    if dd.get("route"):
        route = dd.get("route")
        if route == "decide_route":
            route = "ask_review"
        out["route"] = route
        if route == "extract_tasks":
            # Run extraction inline inside the router node to guarantee progress
            try:
                transcript_text = state.get("transcript") or ""
                if transcript_text:
                    task_list = await extraction.extract_tasks(str(transcript_text))
                    gtasks = [{"assignee": t.owner, "action_item": t.title, "due_date": t.due_date} for t in task_list.tasks]
                    out["tasks"] = gtasks
            except Exception:
                log.exception("decide_route_node: inline extraction failed")
    else:
        # fallback heuristic
        if not state.get("transcript"):
            out["route"] = "ask_transcript"
        elif not state.get("tasks"):
            out["route"] = "extract_tasks"
        else:
            out["route"] = "ask_review"

    return out


# --- BƯỚC 3: ROUTER ---

def router(state: GraphAgentState) -> dict:
    """Deterministic state-machine router for the meeting-tasks-to-Notion pipeline.

    Workflow stages (in order):
      1. collect_transcript  — user pastes / uploads the meeting transcript
      2. extract_tasks       — LLM extracts action items from the transcript
      3. review_tasks (HITL) — user approves or requests revisions to the task list
      4. discover_candidates — agent discovers Notion databases via MCP
      5. pick_database (HITL)— user selects which Notion database to use
      6. resolve_mapping     — agent reads the selected database schema via MCP
      7. confirm_create (HITL)— user confirms the field mapping and task count
      8. create_tasks        — agent creates / updates tasks in Notion via MCP

    Routing is fully deterministic: no LLM call is made here. The `decide_route` node
    is kept as an optional fallback reachable only via an explicit `route_override`.
    """
    # Consume any explicit override (set by action callbacks or fallback LLM)
    if state.get("route_override"):
        route = state.get("route_override")
        return {"route": route, "route_override": None}

    transcript = state.get("transcript") or ""
    tasks      = state.get("tasks") or []
    reviewed   = state.get("reviewed")          # True once user approves task list
    candidates = state.get("candidates")        # None = not yet discovered; [] = none found
    chosen_id  = state.get("chosen_candidate_id")
    mapping    = state.get("mapping")
    decision   = state.get("decision")
    last       = (state.get("last_user_input") or "").strip()

    # ── Stage 1: need meeting transcript ────────────────────────────────────────
    if not transcript:
        return {"route": "ask_transcript"}

    # ── Stage 2: tasks not yet extracted from transcript ────────────────────────
    if not tasks:
        return {"route": "extract_tasks"}

    # ── Stage 3: human review / revision of extracted tasks (HITL) ─────────────
    if not reviewed:
        if last:                          # user typed a response → process it
            return {"route": "process_review"}
        return {"route": "ask_review"}    # show task list and wait for input

    # ── Stage 4: discover available Notion databases via MCP ────────────────────
    if candidates is None:
        return {"route": "discover_candidates"}

    # ── Stage 5: human picks the target Notion database (HITL) ──────────────────
    if not chosen_id:
        if last:
            return {"route": "process_pick_candidate"}
        return {"route": "ask_pick_candidate"}

    # ── Stage 6: resolve Notion database property mapping via MCP ───────────────
    if not mapping:
        return {"route": "resolve_mapping"}

    # ── Stage 7: human confirms field mapping & task creation (HITL) ────────────
    if not decision:
        if last:
            return {"route": "process_confirm"}
        return {"route": "confirm_create"}

    # ── Stage 8: create / update tasks in Notion via MCP ────────────────────────
    if decision == "confirm":
        return {"route": "create_tasks"}
    if decision == "edit":
        # User wants to revise tasks — reset reviewed flag and go back to review
        return {"route": "ask_review"}
    # cancelled
    return {"route": "__end__"}


# --- BƯỚC 4: XÂY GRAPH ---

memory = MemorySaver()
log.info("agent_graph: checkpointer instance=%s", id(memory))
workflow = StateGraph(GraphAgentState)

# register nodes
workflow.add_node("ask_transcript", ask_transcript)
workflow.add_node("extract_tasks", extract_tasks_node)
workflow.add_node("ask_review", ask_review)
workflow.add_node("process_review", process_review)
workflow.add_node("discover_candidates", discover_candidates_node)
workflow.add_node("ask_pick_candidate", ask_pick_candidate)
workflow.add_node("process_pick_candidate", process_pick_candidate)
workflow.add_node("resolve_mapping", resolve_mapping_node)
workflow.add_node("confirm_create", confirm_create_node)
workflow.add_node("process_confirm", process_confirm)
workflow.add_node("create_tasks", create_tasks_node)
workflow.add_node("decide_route", decide_route_node)
workflow.add_node("router", router)

workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router",
    lambda d: d.get("route") if isinstance(d, dict) else None,
    {
        "ask_transcript": "ask_transcript",
        "extract_tasks": "extract_tasks",
        "ask_review": "ask_review",
        "process_review": "process_review",
        "discover_candidates": "discover_candidates",
        "ask_pick_candidate": "ask_pick_candidate",
        "process_pick_candidate": "process_pick_candidate",
        "resolve_mapping": "resolve_mapping",
        "confirm_create": "confirm_create",
        "process_confirm": "process_confirm",
        "create_tasks": "create_tasks",
        "decide_route": "decide_route",
        "__end__": END,
    },
)

# Also support conditional routing when the LLM `decide_route` node returns a route
workflow.add_conditional_edges(
    "decide_route",
    lambda d: d.get("route") if isinstance(d, dict) else None,
    {
        "ask_transcript": "ask_transcript",
        "extract_tasks": "extract_tasks",
        "ask_review": "ask_review",
        "process_review": "process_review",
        "discover_candidates": "discover_candidates",
        "ask_pick_candidate": "ask_pick_candidate",
        "process_pick_candidate": "process_pick_candidate",
        "resolve_mapping": "resolve_mapping",
        "confirm_create": "confirm_create",
        "process_confirm": "process_confirm",
        "create_tasks": "create_tasks",
        "decide_route": "decide_route",
        "__end__": END,
    },
)

async def drive_graph(state, user_text: str | None = None):
    """Run the compiled StateGraph, perform external tool calls when the graph sets `to_call`,
    and present human prompts when the graph asks for input.
    This keeps the graph as the decision-maker while Chainlit handles I/O and side-effects.
    """
    # Accumulate conversation history in session state
    if user_text:
        state.chat_messages = (state.chat_messages or []) + [HumanMessage(content=user_text)]

    # Build initial graph input from session state
    graph_input: dict = {
        "messages": list(state.chat_messages or []),
        "transcript": state.transcript or (user_text if user_text else None),
        "tasks": [
            {"assignee": t.owner, "action_item": t.title, "due_date": t.due_date} for t in (state.tasks or [])
        ],
        "notion_page_ids": state.notion_data_source_id and [state.notion_data_source_id] or None,
        "error_message": None,
        "awaiting_input": False,
        "prompt": None,
        "last_user_input": user_text,
        "to_call": None,
        "candidates": state.graph_candidates,
        "chosen_candidate_id": state.notion_data_source_id,
        "mapping": state.property_mapping.model_dump() if state.property_mapping and hasattr(state.property_mapping, "model_dump") else (state.property_mapping or None),
        "existing_page_ids": state.existing_page_ids or None,
    }

    # Ensure a thread id for the checkpointer
    thread_id = getattr(state, "graph_thread_id", None) or str(uuid4())
    state.graph_thread_id = thread_id
    config = cast(RunnableConfig, {"configurable": {"thread_id": thread_id}})
    log.debug("drive_graph: invoking compiled graph with thread_id=%s checkpointer_id=%s", thread_id, id(memory))

    async def _render_placeholder_content(graph_in: dict, graph_out: dict | None = None) -> str:
        """Render a short inbox/placeholder summary for the UI."""
        lines = []
        lines.append("📥 **Inbox — Meeting Tasks Agent**")
        lines.append(f"- Stage: **{getattr(state, 'stage', 'unknown')}**")
        lines.append(f"- Thread: `{state.graph_thread_id}`")
        tasks_count = len(state.tasks or [])
        lines.append(f"- Tasks: **{tasks_count}**")
        if graph_out:
            to_call = graph_out.get("to_call")
            awaiting = bool(graph_out.get("awaiting_input"))
            prompt = graph_out.get("prompt") or graph_in.get("prompt")
            lines.append(f"- Step: **{to_call or ('awaiting' if awaiting else 'idle')}**")
            if prompt:
                # keep prompt short
                p = str(prompt)
                if len(p) > 200:
                    p = p[:197] + "..."
                lines.append(f"- Prompt: {p}")
        else:
            lines.append("- Step: (starting)")
        return "\n".join(lines)

    async def _upsert_placeholder(graph_in: dict, graph_out: dict | None = None):
        content = await _render_placeholder_content(graph_in, graph_out)
        if getattr(state, "graph_placeholder_id", None):
            try:
                await cl.Message(content=content, id=state.graph_placeholder_id).update()
                return
            except Exception:
                log.debug("Failed to update existing placeholder, will re-create")
        msg = cl.Message(content=content)
        await msg.send()
        state.graph_placeholder_id = msg.id

    def _apply_state_updates(updates: dict, graph_in: dict):
        """Apply a dict of state updates returned by a graph node.
        This updates both the local `graph_in` used for the compiled graph and the
        persistent `state` stored in the Chainlit session where appropriate.
        Supported keys: transcript, tasks (list of GraphTask), candidates, chosen_candidate_id,
        mapping, existing_page_ids, awaiting_input, prompt, decision, results, error_message.
        """
        if not updates or not isinstance(updates, dict):
            return
        # Map GraphAgentState keys to SessionState attributes when names differ
        key_map = {
            "chosen_candidate_id": "notion_data_source_id",
            "candidates": "graph_candidates",
            "mapping": "property_mapping",
            "existing_page_ids": "existing_page_ids",
            "prompt": "graph_prompt",
            "awaiting_input": "awaiting_input",
            "route_override": "graph_route_override",
        }

        for k, v in updates.items():
            try:
                # Tasks need conversion from GraphTask -> Task model
                if k == "tasks":
                    graph_in["tasks"] = [
                        {"assignee": gt.get("assignee"), "action_item": gt.get("action_item"), "due_date": gt.get("due_date")}
                        for gt in (v or [])
                    ]
                    # update session state.tasks as Pydantic Task objects
                    try:
                        state.tasks = [Task(title=gt.get("action_item") or "", description="", owner=gt.get("assignee"), due_date=gt.get("due_date")) for gt in (v or [])]
                    except Exception:
                        log.exception("Failed to convert graph tasks to session Task objects")
                    continue

                # Simple passthrough into graph input
                graph_in[k] = v

                # Sync to session state when there's a mapped attribute
                if k in key_map:
                    setattr(state, key_map[k], v)
                elif hasattr(state, k):
                    setattr(state, k, v)
            except Exception:
                log.exception("_apply_state_updates: failed to apply key %s", k)

    # Initial placeholder
    await _upsert_placeholder(graph_input, None)

    # Ensure graph_output is always defined for later syncing
    graph_output: dict = {}

    try:
        prev_graph_output = None
        repeat_same = 0
        for _ in range(20):
            graph_output = await app.ainvoke(cast(GraphAgentState, graph_input), config=config)
            # Sync key fields from graph_output back to graph_input AND session state immediately
            # so that the next loop iteration (or early return) always has up-to-date state.
            # This prevents stale graph_input from triggering redundant work (e.g. double extraction).
            if graph_output.get("tasks"):
                graph_input["tasks"] = graph_output["tasks"]
                try:
                    state.tasks = [
                        Task(title=gt.get("action_item", ""), description="",
                             owner=gt.get("assignee"), due_date=gt.get("due_date"))
                        for gt in graph_output["tasks"]
                    ]
                except Exception:
                    log.exception("drive_graph: failed to sync tasks from graph_output to session")
            # ── Sync key fields: graph_output → graph_input AND session state ──────────────────────────
            # Persisting to session state is CRITICAL: it ensures the next drive_graph call
            # (e.g. from on_hitl_btn) rebuilds graph_input with the correct values rather
            # than overriding the LangGraph checkpoint with stale None values.
            _session_sync_map = {
                "chosen_candidate_id": "notion_data_source_id",
                "candidates": "graph_candidates",
                "transcript": "transcript",
                "mapping": "property_mapping",
                "existing_page_ids": "existing_page_ids",
            }
            for _sync_key in ("transcript", "candidates", "chosen_candidate_id", "mapping", "existing_page_ids"):
                _sync_val = graph_output.get(_sync_key)
                if _sync_val is not None:
                    graph_input[_sync_key] = _sync_val
                    _sattr = _session_sync_map.get(_sync_key)
                    if _sattr:
                        try:
                            setattr(state, _sattr, _sync_val)
                        except Exception:
                            log.debug("drive_graph: failed to persist %s to session", _sync_key)
            # Sync fields that processing nodes may explicitly clear to None.
            # Must use key-presence check (not truthiness) to propagate None values.
            for _nullable_key in ("last_user_input", "reviewed", "decision"):
                if _nullable_key in (graph_output or {}):
                    graph_input[_nullable_key] = graph_output[_nullable_key]
            # Reflect updated history in graph_input for subsequent iterations
            graph_input["messages"] = list(state.chat_messages or [])
            # detect repeated identical outputs to avoid infinite retry loops
            if graph_output == prev_graph_output:
                repeat_same += 1
            else:
                prev_graph_output = graph_output
                repeat_same = 0
            if repeat_same >= 2:
                log.warning("drive_graph: detected repeated graph_output; applying fallback")
                # If we have a transcript but no tasks, proactively run extraction here
                try:
                    if graph_input.get("transcript") and not (graph_input.get("tasks") or []):
                        log.info("drive_graph: fallback -> calling extraction.extract_tasks directly")
                        try:
                            task_list = await extraction.extract_tasks(str(graph_input.get("transcript") or ""))
                        except LLMResponseError as e:
                            await cl.Message(content=f"⚠️ Couldn't extract tasks: {e}").send()
                            return
                        gtasks = [{"assignee": t.owner, "action_item": t.title, "due_date": t.due_date} for t in task_list.tasks]
                        graph_input["tasks"] = gtasks
                        state.tasks = task_list.tasks
                        # reset repeat detection and continue processing
                        prev_graph_output = None
                        repeat_same = 0
                        continue
                except Exception:
                    log.exception("drive_graph: fallback extraction failed")
                    return
            log.debug("drive_graph: graph_output=%s", graph_output)
            # update placeholder with latest graph state
            try:
                await _upsert_placeholder(graph_input, graph_output)
            except Exception:
                log.exception("Failed to upsert placeholder message")

            # If a node returned explicit state updates, apply them and re-run the graph
            state_updates = graph_output.get("state_updates") or graph_output.get("update_state")
            if isinstance(state_updates, dict):
                log.debug("drive_graph: applying state_updates=%s", state_updates)
                try:
                    _apply_state_updates(state_updates, graph_input)
                    # reflect changes in the placeholder immediately
                    try:
                        await _upsert_placeholder(graph_input, None)
                    except Exception:
                        log.exception("Failed to refresh placeholder after state_updates")
                except Exception:
                    log.exception("drive_graph: failed applying state_updates")
                # Re-run graph iteration with updated graph_input
                continue

            # If the graph asks for human input, present a prompt (special-case candidates)
            if graph_output.get("awaiting_input"):
                prompt = graph_output.get("prompt") or "Please respond."
                hitl = graph_output.get("hitl_interaction") or {}
                buttons = hitl.get("buttons") or []
                allow_freetext: bool = hitl.get("allow_freetext", True)

                # Record the AI prompt in conversation history so the next turn has full context.
                state.chat_messages = (state.chat_messages or []) + [AIMessage(content=prompt)]

                # Store awaiting flag in session so on_message knows to pass user_text to drive_graph.
                try:
                    state.awaiting_input = True
                    state.graph_prompt = prompt
                    state.graph_candidates = None  # candidates now live in hitl_interaction
                except Exception:
                    log.exception("Failed to set awaiting_input session flags")

                # Build Chainlit action objects from the hitl_interaction button schema.
                cl_actions = [
                    cl.Action(
                        name="hitl_btn",
                        payload={"value": btn.get("value", "")},
                        label=btn.get("label") or btn.get("value", ""),
                    )
                    for btn in buttons
                    if btn.get("label") or btn.get("value")
                ]

                try:
                    # Use cl.Message for all HITL prompts.
                    # cl.AskActionMessage used with asyncio.create_task (non-awaited) can
                    # consume button-click events before the @cl.action_callback handler sees
                    # them, causing the UI to freeze. cl.Message + action callbacks is the
                    # correct non-blocking pattern for this architecture.
                    asyncio.create_task(
                        cl.Message(content=prompt, actions=cl_actions).send()
                    )
                except Exception:
                    log.exception("Failed to send HITL interaction message")
                    await cl.Message(content=prompt).send()
                return

            # If the graph requests an external call, perform it here and feed the result back in
            if graph_output.get("to_call"):
                call = graph_output.get("to_call")
                log.debug("drive_graph: to_call=%s", call)

                if call == "extract_tasks":
                    if not graph_input.get("transcript"):
                        # Let the graph/agent prompt the user instead of sending a default message
                        try:
                            state.awaiting_input = True
                            state.graph_prompt = "Please paste the meeting transcript or attach a file."
                            # Use a plain message so the composer allows attaching files.
                            asyncio.create_task(cl.Message(content=state.graph_prompt).send())
                        except Exception:
                            log.exception("Failed to send prompt message for missing transcript in extract_tasks")
                        return
                    try:
                        task_list = await extraction.extract_tasks(str(graph_input.get("transcript") or ""))
                    except LLMResponseError as e:
                        await cl.Message(content=f"⚠️ Couldn't extract tasks: {e}").send()
                        return
                    # If no tasks were extracted, ask the user for more transcript or allow cancelling.
                    if not getattr(task_list, "tasks", None):
                        # Present a non-blocking AskUserMessage so the user can provide
                        # more transcript; on_message will resume the graph.
                        try:
                            state.awaiting_input = True
                            state.graph_prompt = (
                                "I couldn't find any action items in that transcript.\n\n"
                                "Please paste more transcript text, or reply 'none' to stop trying."
                            )
                            asyncio.create_task(cl.Message(content=state.graph_prompt).send())
                        except Exception:
                            log.exception("Failed to send prompt message after empty extraction")
                        return

                    gtasks = [{"assignee": t.owner, "action_item": t.title, "due_date": t.due_date} for t in task_list.tasks]
                    graph_input["tasks"] = gtasks
                    graph_input["last_user_input"] = None  # clear so transcript isn't mistaken for review input
                    state.tasks = task_list.tasks
                    continue

                if call == "decide_route":
                    # Ask the LLM to choose the next route or produce a conversational reply.
                    try:
                        runnable = FptStructuredRunnable(RouteDecision)
                        # Summarize current graph state for the LLM
                        tasks = graph_input.get("tasks") or []
                        tasks_preview = "\n".join([f"- {t.get('action_item')} (owner: {t.get('assignee')})" for t in tasks]) or "(no tasks)"
                        # Build conversation history context
                        _hist_msgs = graph_input.get("messages") or []
                        if _hist_msgs:
                            _hist_lines = []
                            for _hm in _hist_msgs[-8:]:
                                _hrole = "User" if isinstance(_hm, HumanMessage) else "Assistant"
                                _hist_lines.append(f"{_hrole}: {str(_hm.content)[:300]}")
                            _chat_ctx = "\n".join(_hist_lines)
                        else:
                            _chat_ctx = "(no prior conversation)"
                        user_msg = (
                            "Analyze the following Meeting Tasks Agent state and pick the best next route from:"
                            " ask_transcript, extract_tasks, ask_review, process_review, discover_candidates, ask_pick_candidate,"
                            " process_pick_candidate, resolve_mapping, confirm_create, process_confirm, create_tasks, __end__.\n\n"
                            f"Recent conversation:\n{_chat_ctx}\n\n"
                            f"State summary:\n- Transcript present: {bool(graph_input.get('transcript'))}\n- Transcript length: {len(str(graph_input.get('transcript') or ''))}\n- Last user input: {graph_input.get('last_user_input')!r}\n- Tasks ({len(tasks)}):\n{tasks_preview}\n- Candidates: {graph_input.get('candidates') or []}\n- Chosen candidate id: {graph_input.get('chosen_candidate_id') or None}\n- Mapping present: {bool(graph_input.get('mapping'))}\n\n"
                            "Return ONLY a JSON object matching this schema: {\n  \"route\": string|null,\n  \"prompt\": string|null,\n  \"awaiting_input\": boolean|null,\n  \"state_updates\": object|null,\n  \"reply\": string|null\n}\n\n"
                            "Guidelines: If you want the agent to ask the human for input, set 'awaiting_input' true and provide 'prompt'."
                        )
                        llm_messages = [SystemMessage(content="You are a concise routing assistant for a meeting task workflow."), HumanMessage(content=user_msg)]
                        decision = await runnable.ainvoke(llm_messages)
                    except LLMResponseError as e:
                        log.exception("decide_route LLM failed")
                        await cl.Message(content=f"⚠️ Routing assistant failed: {e}").send()
                        return

                    # decision is a BaseModel instance (RouteDecision)
                    # Convert Pydantic result to a plain dict
                    from typing import cast as _cast
                    dd = _cast(dict, decision.model_dump())

                    # Apply any state updates the LLM returned
                    if dd.get("state_updates"):
                        try:
                            _apply_state_updates(dd.get("state_updates") or {}, graph_input)
                            try:
                                await _upsert_placeholder(graph_input, None)
                            except Exception:
                                log.exception("Failed to refresh placeholder after router state_updates")
                        except Exception:
                            log.exception("Failed applying router state_updates")

                    # If the LLM supplied a conversational reply, send it
                    if dd.get("reply"):
                        try:
                            reply_text = _cast(str, dd.get("reply") or "")
                            await cl.Message(content=reply_text).send()
                        except Exception:
                            log.exception("Failed to send router reply message")

                    # If LLM expects further input from user, set awaiting flags and present prompt
                    if dd.get("awaiting_input"):
                        try:
                            state.awaiting_input = True
                            state.graph_prompt = dd.get("prompt") or "Please respond."
                            state.graph_candidates = dd.get("state_updates", {}).get("candidates") if dd.get("state_updates") else None
                        except Exception:
                            log.exception("Failed to set awaiting_input from router decision")
                        # Present prompt to the user via AskUserMessage so the graph/agent UI is consistent
                        try:
                            asyncio.create_task(cl.AskUserMessage(content=(dd.get("prompt") or "Please respond."), timeout=300).send())
                        except Exception:
                            log.exception("Failed to send AskUserMessage for router prompt")
                        return

                    # If LLM returned a concrete route, set an override and re-run the graph
                    if dd.get("route"):
                        try:
                            # Use state_updates helper so session state syncs as well
                            _apply_state_updates({"route_override": dd.get("route")}, graph_input)
                            try:
                                await _upsert_placeholder(graph_input, None)
                            except Exception:
                                log.exception("Failed to refresh placeholder after setting route_override")
                        except Exception:
                            log.exception("Failed to set route_override from router decision")
                        # Re-evaluate the graph with updated graph_input
                        continue

                    # No actionable output from router
                    await cl.Message(content="Routing assistant didn't return a route or reply.").send()
                    return

                if call == "revise_tasks":
                    feedback = str(graph_output.get("revise_feedback") or graph_input.get("last_user_input") or "")
                    try:
                        current = [Task(title=gt["action_item"], description="", owner=gt.get("assignee"), due_date=gt.get("due_date")) for gt in (graph_input.get("tasks") or [])]
                        task_list = await extraction.revise_tasks(TaskList(tasks=current), feedback)
                    except Exception as e:
                        log.exception("revise_tasks failed")
                        await cl.Message(content=f"⚠️ Couldn't revise tasks: {e}").send()
                        return
                    graph_input["tasks"] = [{"assignee": t.owner, "action_item": t.title, "due_date": t.due_date} for t in task_list.tasks]
                    graph_input["last_user_input"] = None  # feedback consumed — prevent re-routing to process_review
                    graph_input["reviewed"] = None          # force re-review of revised tasks
                    state.tasks = task_list.tasks
                    continue

                if call == "discover_data_sources":
                    task_titles = ", ".join(t.title for t in (state.tasks or []))
                    try:
                        candidates_result = await notion_mapping.discover_data_sources(state, task_titles)
                    except LLMResponseError as e:
                        log.exception("discover_data_sources failed")
                        await cl.Message(content=f"⚠️ Couldn't discover a Notion database: {e}").send()
                        return
                    candidates = [{"id": c.id, "name": c.name, "url": getattr(c, "url", None)} for c in candidates_result.candidates]
                    graph_input["candidates"] = candidates
                    graph_input["last_user_input"] = None  # clear so "looks good" isn't mistaken for a DB pick
                    state.graph_candidates = candidates
                    continue

                if call == "resolve_property_mapping":
                    dsid = graph_input.get("chosen_candidate_id") or state.notion_data_source_id
                    if not dsid:
                        await cl.Message(content="No data source id to resolve mapping for.").send()
                        return
                    try:
                        mapping = await notion_mapping.resolve_property_mapping(state, dsid)
                    except Exception as e:
                        log.exception("resolve_property_mapping failed")
                        await cl.Message(content=f"⚠️ Couldn't read the Notion database schema: {e}").send()
                        return
                    graph_input["mapping"] = mapping.model_dump() if hasattr(mapping, "model_dump") else mapping
                    state.property_mapping = mapping
                    continue

                if call == "create_tasks_in_notion":
                    dsid = graph_input.get("chosen_candidate_id") or state.notion_data_source_id
                    if not dsid or not state.property_mapping:
                        await cl.Message(content="Missing data source or property mapping to create tasks.").send()
                        return
                    tasks_to_create = [Task(title=gt["action_item"], description="", owner=gt.get("assignee"), due_date=gt.get("due_date")) for gt in (graph_input.get("tasks") or [])]
                    try:
                        results = await notion_mapping.create_tasks_in_notion(state, dsid, state.property_mapping, tasks_to_create, state.existing_page_ids)
                    except LLMResponseError as e:
                        log.exception("create_tasks_in_notion failed")
                        await cl.Message(content=f"⚠️ Couldn't create/update tasks in Notion: {e}").send()
                        return
                    lines = []
                    for r in results:
                        icon = "✅" if r.ok else "❌"
                        label = r.action
                        lines.append(f"{icon} {r.task_title} — {label}")
                    final_content = "**Done:**\n\n" + "\n".join(lines)
                    # Replace the placeholder with the final result (or send new if missing)
                    try:
                        if getattr(state, "graph_placeholder_id", None):
                            await cl.Message(content=final_content, id=state.graph_placeholder_id).update()
                            # clear placeholder id so next run creates a fresh one
                            state.graph_placeholder_id = None
                        else:
                            await cl.Message(content=final_content).send()
                    except Exception:
                        log.exception("Failed to update placeholder with final result; sending as new message")
                        await cl.Message(content=final_content).send()
                    state.write_confirmed = True
                    state.stage = "done"
                    return

                log.warning("drive_graph: unknown to_call=%s", call)
                return

        # Nothing left to do: sync returned state back to session and show prompt if any
        if graph_output and isinstance(graph_output, dict):
            state.transcript = graph_output.get("transcript") or state.transcript
            if graph_output.get("tasks") is not None:
                state.tasks = [Task(title=gt["action_item"], description="", owner=gt.get("assignee"), due_date=gt.get("due_date")) for gt in graph_output.get("tasks", [])]
            if graph_output.get("mapping") is not None and not state.property_mapping:
                try:
                    state.property_mapping = graph_output.get("mapping")
                except Exception:
                    state.property_mapping = graph_output.get("mapping")
            if graph_output.get("prompt"):
                prompt_text = graph_output.get("prompt") or ""
                await cl.Message(content=prompt_text).send()
        return
    finally:
        # Clean up placeholder on exit if still present
        try:
            if getattr(state, "graph_placeholder_id", None):
                try:
                    await cl.Message(content="", id=state.graph_placeholder_id).remove()
                except Exception:
                    log.debug("Failed to remove placeholder message during cleanup")
                state.graph_placeholder_id = None
        except Exception:
            log.exception("Error during placeholder cleanup in drive_graph")

# All nodes end the immediate step; router will re-evaluate on the next invoke
for n in [
    "ask_transcript",
    "extract_tasks",
    "ask_review",
    "process_review",
    "discover_candidates",
    "ask_pick_candidate",
    "process_pick_candidate",
    "resolve_mapping",
    "confirm_create",
    "process_confirm",
    "create_tasks",
]:
    workflow.add_edge(n, END)

app = workflow.compile(checkpointer=memory)


# --------------------
# Chainlit handlers moved here from app.py
# --------------------

WELCOME = """**Meeting Tasks Agent**

Paste a meeting transcript below (or attach a `.txt`/`.md` file) and I'll pull out the action items — owner, due date, and dependencies — for you to review.

When you're ready to create the tasks in Notion, connect Notion's hosted MCP server via the 🔌 icon in the composer: add a `stdio` server with command `npx -y mcp-remote https://mcp.notion.com/mcp`. The first time, it'll open a browser for you to sign in to Notion (OAuth) — after that it reconnects automatically. You can do this any time before confirming."""


def _render_task_list(tasks: list) -> str:
    if not tasks:
        return "_No action items found._"
    lines = []
    for i, t in enumerate(tasks, 1):
        # Support both Pydantic Task objects and plain dicts (GraphTask)
        if isinstance(t, dict):
            title        = (t.get("title") or t.get("action_item") or "").strip()
            description  = (t.get("description") or "").strip()
            owner        = t.get("owner") or t.get("assignee")
            due_date     = t.get("due_date")
            status       = t.get("status")
            dependencies = t.get("dependencies") or []
        else:
            title        = (getattr(t, "title", None) or "").strip()
            description  = (getattr(t, "description", None) or "").strip()
            owner        = getattr(t, "owner", None)
            due_date     = getattr(t, "due_date", None)
            status       = getattr(t, "status", None)
            dependencies = getattr(t, "dependencies", None) or []

        # Title row
        line = f"**{i}. {title or '*(no title)*'}**"

        # Metadata row (only the fields that are set)
        meta = []
        if owner:
            meta.append(f"👤 {owner}")
        if due_date:
            meta.append(f"📅 {due_date}")
        if status:
            meta.append(f"⏳ {status}")
        if meta:
            line += f"\n   *{' · '.join(meta)}*"

        # Optional description
        if description:
            line += f"\n   {description}"

        # Dependencies
        if dependencies:
            dep_str = ", ".join(dependencies) if isinstance(dependencies, list) else str(dependencies)
            line += f"\n   ↪ *Depends on: {dep_str}*"

        lines.append(line)
    return "\n\n".join(lines)


async def _extract_from_message(message: cl.Message) -> str | None:
    # Collect elements — always log count at INFO so upload issues are visible without DEBUG
    try:
        elems = list(message.elements)
    except Exception:
        elems = []
    log.info("_extract_from_message: content_len=%d elements=%d", len(message.content or ""), len(elems))

    for element in elems:
        # Log element attributes to help diagnose upload issues
        try:
            attrs = {a: getattr(element, a) for a in ("path", "content", "name", "chainlit_key", "url", "mime") if hasattr(element, a)}
            log.info("_extract_from_message: element attrs=%s", attrs)
        except Exception:
            pass

        # Raw bytes content (older Chainlit variants)
        data_blob = getattr(element, "data", None)
        if data_blob and isinstance(data_blob, (bytes, bytearray)):
            try:
                return unicodedata.normalize("NFC", bytes(data_blob).decode("utf-8", errors="replace"))
            except Exception:
                pass

        # Local filesystem path — most common case for spontaneous file uploads
        path = getattr(element, "path", None)
        if path:
            try:
                with open(path, "rb") as f:
                    raw = f.read()
                log.info("_extract_from_message: read %d bytes from element.path=%s", len(raw), path)
                return unicodedata.normalize("NFC", raw.decode("utf-8", errors="replace"))
            except Exception:
                log.exception("_extract_from_message: failed to read element.path=%s", path)

        # Inline text content
        content = getattr(element, "content", None)
        if content and isinstance(content, str) and content.strip():
            return unicodedata.normalize("NFC", content)

        # Fallback: resolve via chainlit_key → session.files (covers path-less elements)
        chainlit_key = getattr(element, "chainlit_key", None)
        if chainlit_key:
            try:
                from chainlit import context as _cl_ctx
                _fd = _cl_ctx.session.files.get(chainlit_key)
                if _fd and _fd.get("path"):
                    with open(str(_fd["path"]), "rb") as _f:
                        raw = _f.read()
                    log.info("_extract_from_message: read %d bytes via chainlit_key=%s", len(raw), chainlit_key)
                    return unicodedata.normalize("NFC", raw.decode("utf-8", errors="replace"))
            except Exception:
                log.exception("_extract_from_message: chainlit_key fallback failed for key=%s", chainlit_key)

    # Plain-text message content (no file attachment)
    if message.content and message.content.strip():
        return unicodedata.normalize("NFC", message.content)

    # Last-resort fallback: scan session.files for the most recently uploaded text file that
    # hasn't been processed yet.  This handles cases where message.elements is empty because
    # the file reference lookup in Chainlit's emitter failed (e.g., after a hot-reload that
    # cleared in-memory session state, or a frontend/backend version mismatch).
    try:
        from chainlit import context as _cl_ctx
        _session = _cl_ctx.session
        _done_ids: set = cl.user_session.get("_used_file_ids") or set()
        # Iterate newest-first (dict insertion order, reversed)
        for _fid, _fd in reversed(list(getattr(_session, "files", {}).items())):
            if _fid in _done_ids:
                continue
            _mime: str = _fd.get("type", "") or ""
            _fname: str = _fd.get("name", "") or ""
            _is_text = "text" in _mime or _fname.lower().endswith((".txt", ".md", ".rst"))
            if not _is_text:
                continue
            _fpath = _fd.get("path")
            if not _fpath:
                continue
            try:
                with open(str(_fpath), "rb") as _f:
                    _raw = _f.read()
                _text = unicodedata.normalize("NFC", _raw.decode("utf-8", errors="replace")).strip()
                if _text:
                    log.info("_extract_from_message: session.files fallback read %r (%d bytes)", _fname, len(_raw))
                    cl.user_session.set("_used_file_ids", _done_ids | {_fid})
                    return _text
            except Exception:
                log.exception("_extract_from_message: session.files fallback failed reading %r", _fpath)
    except Exception:
        log.exception("_extract_from_message: session.files fallback lookup failed")

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
        try:
            elems = list(message.elements)
        except Exception:
            elems = []
        if elems:
            await cl.Message(
                content="I couldn't read the uploaded file. Please attach a plain **.txt** or **.md** file, or paste the transcript as text."
            ).send()
        else:
            await cl.Message(
                content="Please paste the meeting transcript below (or attach a `.txt`/`.md` file)."
            ).send()
        return

    state.transcript = transcript
    thinking = cl.Message(content="⏳ Extracting action items from the transcript…")
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
    log.info("stage: collecting_transcript → reviewing_tasks, %d task(s)", len(state.tasks))

    if state.tasks:
        thinking.content = (
            f"**Extracted {len(state.tasks)} action item(s):**\n\n{_render_task_list(state.tasks)}\n\n"
            "💬 **You can refine this list by chatting**, for example:\n"
            "- *\"Task 2 owner should be Alice\"*\n"
            "- *\"Merge tasks 1 and 3 into one\"*\n"
            "- *\"Remove task 4 \u2014 it's already done\"*\n\n"
            "When the list looks right, click **Looks good** to save to Notion."
        )
        thinking.actions = [
            cl.Action(name="proceed_to_confirm", payload={}, label="✅ Looks good, proceed to Notion"),
        ]
    else:
        thinking.content = (
            "I couldn't find any action items in that transcript.\n\n"
            "Please paste more context or a more detailed transcript."
        )
        thinking.actions = []
    await thinking.update()


async def _handle_reviewing_tasks(message: cl.Message, state):
    feedback = (message.content or "").strip()
    if not feedback:
        return
    thinking = cl.Message(content="⏳ Revising task list…")
    await thinking.send()
    try:
        task_list = await extraction.revise_tasks(TaskList(tasks=state.tasks), feedback)
    except Exception as e:
        log.exception("revise_tasks failed")
        thinking.content = f"⚠️ Couldn't revise tasks: {e}"
        await thinking.update()
        return
    state.tasks = task_list.tasks
    thinking.content = (
        f"**Updated task list ({len(state.tasks)} task(s)):**\n\n{_render_task_list(state.tasks)}\n\n"
        "Keep editing or click **Looks good** when the list is ready."
    )
    if state.tasks:
        thinking.actions = [
            cl.Action(name="proceed_to_confirm", payload={}, label="✅ Looks good, proceed to Notion"),
        ]
    else:
        thinking.actions = []
    await thinking.update()


@cl.action_callback("proceed_to_confirm")
async def on_proceed_to_confirm(action: cl.Action):
    state = get_state()
    if not state.tasks:
        await cl.Message(
            content="No tasks to proceed with. Please paste a transcript containing action items first."
        ).send()
        return
    log.info("proceed_to_confirm: starting _run_confirmation_flow")
    state.stage = "confirming"
    await _run_confirmation_flow()


async def _create_new_db_and_continue(state) -> None:
    """Prompt for a new Notion database name + parent page, create it, then continue the graph.
    Called when the user clicks the '➕ Create a new database' HITL button.
    """
    name_msg = await cl.AskUserMessage(
        content="What should the new database be called?", timeout=300
    ).send()
    if not name_msg or not (name_msg.get("output") or "").strip():
        await cl.Message(content="Cancelled — no database name provided.").send()
        return
    new_db_name = (name_msg.get("output") or "").strip()

    parent_msg = await cl.AskUserMessage(
        content="Which Notion page should this new database be created in? (name or URL)",
        timeout=300,
    ).send()
    if not parent_msg or not (parent_msg.get("output") or "").strip():
        await cl.Message(content="Cancelled — no parent page provided.").send()
        return
    parent_page = (parent_msg.get("output") or "").strip()

    creating = cl.Message(content=f"Creating database **{new_db_name}** inside **{parent_page}**…")
    await creating.send()
    try:
        created = await notion_mapping.create_data_source(state, new_db_name, parent_page)
    except Exception as e:
        log.exception("_create_new_db_and_continue: create_data_source failed")
        creating.content = f"⚠️ Couldn't create the database: {e}"
        await creating.update()
        return

    creating.content = (
        f"✅ Created database **{created.name}**"
        + (f" — {created.url}" if getattr(created, "url", None) else "")
    )
    await creating.update()

    # Set the chosen database directly in session state so the deterministic router
    # skips Stage 5 (pick database) and advances straight to Stage 6 (resolve mapping).
    state.notion_data_source_id = created.id
    state.notion_data_source_name = getattr(created, "name", new_db_name)
    state.awaiting_input = False
    state.graph_prompt = None
    await drive_graph(state)


@cl.action_callback("hitl_btn")
async def on_hitl_btn(action: cl.Action):
    """Generic handler for all HITL buttons defined by the hitl_interaction schema.

    Each button carries its submit value in action.payload["value"].  That value becomes
    `user_text` for the next drive_graph call so the deterministic router can dispatch to
    the correct processing node (process_review / process_pick_candidate / process_confirm).
    """
    state = get_state()
    payload = getattr(action, "payload", {}) or {}
    value = (payload.get("value") or "").strip()
    if not value:
        return

    state.awaiting_input = False
    state.graph_prompt = None

    # Special case: create-new-database flow (requires multi-step user prompts)
    if value == "__create_new__":
        log.info("hitl_btn: __create_new__ → _create_new_db_and_continue")
        await _create_new_db_and_continue(state)
        return

    log.info("hitl_btn: value=%r → drive_graph", value)
    await drive_graph(state, user_text=value)


@cl.action_callback("pick_data_source")
async def on_pick_data_source(action: cl.Action):
    """Legacy callback kept for backward compatibility. New messages use hitl_btn."""
    state = get_state()
    payload = getattr(action, "payload", {}) or {}
    # Use the database name as the user-text so process_pick_candidate can match it
    value = payload.get("name") or payload.get("id") or ""
    state.awaiting_input = False
    state.graph_prompt = None
    await drive_graph(state, user_text=value)


@cl.action_callback("create_new_data_source")
async def on_create_new_data_source(action: cl.Action):
    """Legacy callback kept for backward compatibility. New messages use hitl_btn with __create_new__."""
    state = get_state()
    state.awaiting_input = False
    state.graph_prompt = None
    await _create_new_db_and_continue(state)


@cl.action_callback("cancel_pick")
async def on_cancel_pick(action: cl.Action):
    """Legacy callback kept for backward compatibility."""
    state = get_state()
    state.awaiting_input = False
    state.graph_prompt = None
    await drive_graph(state, user_text="cancel")


@cl.action_callback("confirm")
async def on_confirm(action: cl.Action):
    """Legacy callback kept for backward compatibility. New messages use hitl_btn."""
    state = get_state()
    state.awaiting_input = False
    state.graph_prompt = None
    await drive_graph(state, user_text="confirm")


@cl.action_callback("edit")
async def on_edit(action: cl.Action):
    """Legacy callback kept for backward compatibility. New messages use hitl_btn."""
    state = get_state()
    state.awaiting_input = False
    state.graph_prompt = None
    await drive_graph(state, user_text="edit")


@cl.action_callback("cancel")
async def on_cancel(action: cl.Action):
    """Legacy callback kept for backward compatibility. New messages use hitl_btn."""
    state = get_state()
    state.awaiting_input = False
    state.graph_prompt = None
    await drive_graph(state, user_text="cancel")


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
            content="Couldn't find any Notion database automatically. Make sure the target database is shared with your Notion connection, then try again."
        ).send()
        state.stage = "reviewing_tasks"
        return

    candidate_lines = "\n".join(f"- **{c.name}**" + (f" — {c.url}" if c.url else "") for c in candidates)
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
        new_db_name = name_msg.get("output", "").strip()

        parent_msg = await cl.AskUserMessage(content="Which Notion page should this new database be created in? (name or URL)", timeout=300).send()
        if not parent_msg or not parent_msg.get("output", "").strip():
            await cl.Message(content="Cancelled. No tasks were created.").send()
            state.stage = "reviewing_tasks"
            return
        parent_page_name = parent_msg.get("output", "").strip()

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
        creating_msg.content = f"Created database **{data_source_name}**" + (f" — {created.url}" if created.url else "")
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
        content=(f"{summary_line} in Notion database **{data_source_name}**.\n\n" + "Field mapping:\n" + "\n".join(mapping_lines)),
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

    try:
        results = await notion_mapping.create_tasks_in_notion(
            state, state.notion_data_source_id, state.property_mapping, state.tasks, state.existing_page_ids
        )
    except LLMResponseError as e:
        log.exception("create_tasks_in_notion failed")
        progress.content = f"⚠️ Couldn't create/update tasks in Notion: {e}"
        await progress.update()
        state.stage = "reviewing_tasks"
        return

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
    stage = getattr(state, "stage", "collecting_transcript")
    log.info("on_message: stage=%s content_len=%d", stage, len(message.content or ""))

    if stage == "reviewing_tasks" and state.transcript:
        # Typed messages in the review stage are task-revision requests
        await _handle_reviewing_tasks(message, state)
    elif stage in ("confirming", "creating", "done"):
        # These stages are driven by button callbacks; text input has no effect
        await cl.Message(
            content="Please use the buttons above to continue, or start a **New Chat** to restart."
        ).send()
    else:
        # collecting_transcript (or any unknown stage): expect a transcript
        await _handle_collecting_transcript(message, state)