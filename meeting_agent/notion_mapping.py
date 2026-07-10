import difflib
import json
import unicodedata

from .mcp_tools import dispatch_tool_call
from .models import Task
from .state import SessionState

# Notion MCP tool names have already changed once (database -> data source
# migration) and may change again, so resolve by fuzzy name match against
# whatever the connected server actually advertises rather than hardcoding.
_TOOL_NAME_CANDIDATES = {
    "retrieve_data_source": ["retrieve-a-data-source", "retrieve-a-database", "get-data-source"],
    "create_page": ["create-a-page", "create-page"],
    "update_page_content": ["update-page-markdown", "update-a-page", "append-block-children"],
}

_OWNER_ALIASES = ["owner", "assignee", "assigned to", "responsible", "phụ trách", "người phụ trách"]
_DUE_DATE_ALIASES = ["due", "due date", "deadline", "due by", "target date", "hạn", "hạn chót"]
_COMPATIBLE_OWNER_TYPES = {"select", "multi_select", "rich_text"}
_COMPATIBLE_DATE_TYPES = {"date"}


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip().lower()


def resolve_tool_name(state: SessionState, purpose: str) -> str | None:
    all_names = [name for tools in state.mcp_tool_cache.values() for name in tools]
    for candidate in _TOOL_NAME_CANDIDATES[purpose]:
        if candidate in all_names:
            return candidate
    matches = difflib.get_close_matches(_TOOL_NAME_CANDIDATES[purpose][0], all_names, n=1, cutoff=0.5)
    return matches[0] if matches else None


async def fetch_data_source_schema(state: SessionState, data_source_id: str) -> dict:
    tool_name = resolve_tool_name(state, "retrieve_data_source")
    if tool_name is None:
        raise RuntimeError("No connected Notion MCP tool found to retrieve a data source schema")
    raw = await dispatch_tool_call(state, tool_name, {"data_source_id": data_source_id})
    data = json.loads(raw)
    return data.get("properties", data)


def _best_alias_match(aliases: list[str], properties: dict, compatible_types: set[str]) -> str | None:
    candidate_names = [name for name, meta in properties.items() if meta.get("type") in compatible_types]
    if not candidate_names:
        return None
    normalized_map = {_norm(name): name for name in candidate_names}
    for alias in aliases:
        if _norm(alias) in normalized_map:
            return normalized_map[_norm(alias)]
    for alias in aliases:
        matches = difflib.get_close_matches(_norm(alias), normalized_map.keys(), n=1, cutoff=0.6)
        if matches:
            return normalized_map[matches[0]]
    return None


def compute_property_mapping(properties: dict) -> dict:
    title_prop = next((name for name, meta in properties.items() if meta.get("type") == "title"), None)
    return {
        "title": title_prop,
        "owner": _best_alias_match(_OWNER_ALIASES, properties, _COMPATIBLE_OWNER_TYPES),
        "due_date": _best_alias_match(_DUE_DATE_ALIASES, properties, _COMPATIBLE_DATE_TYPES),
    }


def _page_body_lines(task: Task, mapping: dict) -> list[str]:
    lines = [task.description] if task.description else []
    if task.owner and mapping.get("owner") is None:
        lines.append(f"Owner: {task.owner}")
    if task.due_date and mapping.get("due_date") is None:
        lines.append(f"Due date: {task.due_date}")
    if task.dependencies:
        lines.append("Depends on: " + ", ".join(task.dependencies))
    if task.source_excerpt:
        lines.append(f"Source: “{task.source_excerpt}”")
    return lines


def build_create_page_args(data_source_id: str, mapping: dict, task: Task) -> dict:
    properties: dict = {}
    if mapping.get("title"):
        properties[mapping["title"]] = {"title": [{"text": {"content": task.title}}]}
    if mapping.get("owner") and task.owner:
        properties[mapping["owner"]] = {"rich_text": [{"text": {"content": task.owner}}]}
    if mapping.get("due_date") and task.due_date:
        properties[mapping["due_date"]] = {"date": {"start": task.due_date}}

    body_text = "\n".join(_page_body_lines(task, mapping))
    return {
        "parent": {"data_source_id": data_source_id},
        "properties": properties,
        "content": body_text,
    }


async def create_tasks_in_notion(
    state: SessionState, data_source_id: str, mapping: dict, tasks: list[Task]
) -> list[dict]:
    tool_name = resolve_tool_name(state, "create_page")
    if tool_name is None:
        raise RuntimeError("No connected Notion MCP tool found to create a page")

    results = []
    for task in tasks:
        args = build_create_page_args(data_source_id, mapping, task)
        raw = await dispatch_tool_call(state, tool_name, args)
        ok = not raw.startswith("ERROR:")
        results.append({"task": task.title, "ok": ok, "detail": raw})
    return results
