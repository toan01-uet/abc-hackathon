import re

from langchain_core.messages import HumanMessage

from . import mcp_tools
from .agent import build_notion_agent, run_agent
from .langchain_llm import FptStructuredRunnable, LLMResponseError
from .logging_config import get_logger
from .models import Task
from .notion_models import (
    CreatePageOutcome,
    DataSourceCandidates,
    ExistingTaskMatches,
    NewDataSource,
    NotionCreatePagesArgs,
    NotionUpdatePageArgs,
    PropertyMapping,
)
from .state import SessionState

log = get_logger("notion_mapping")

# Real Notion hosted-MCP tool names (verified via live session.list_tools()) —
# no fuzzy name matching, these are exact and stable for this server.
_READ_ONLY_DISCOVERY_TOOLS = {"notion-search", "notion-fetch", "notion-query-data-sources"}
_CREATE_DATABASE_TOOLS = {"notion-search", "notion-create-database"}
_CREATE_PAGES_TOOL = "notion-create-pages"
_UPDATE_PAGE_TOOL = "notion-update-page"

_TOOL_SEMANTICS = """You have these real Notion MCP tools available:
- notion-search: full-text search across the user's Notion workspace. Args: {"query": str, "query_type": "internal"|"user", "page_size": int}.
- notion-fetch: given an id or URL, returns the page/database as a JSON object with a "text" field containing "enhanced markdown". Databases show their data sources embedded as <data-source url="collection://..."> tags — that collection://<id> IS the data source id to use later as a parent for notion-create-pages. The schema appears as a JSON blob under "schema" inside <data-source-state>, e.g. {"Name": {"type": "title"}, "Status": {"type": "select"}, ...} — read property names and types from there.
- notion-query-data-sources: SQL-like querying over a data source's rows, using collection://<id> as the table name."""

_DISCOVERY_SYSTEM_PROMPT = f"""You are helping find candidate Notion pages/databases where a
user's extracted meeting tasks could be created.

{_TOOL_SEMANTICS}

Use these tools to find plausible target Notion databases (data sources) for creating task
pages. Consider anything plausible — don't narrow to just one. When you have enough
information, stop calling tools and describe every plausible candidate you found."""

_CREATE_DB_SYSTEM_PROMPT = f"""You are helping create a brand-new Notion database for tracking meeting tasks.

{_TOOL_SEMANTICS}
- notion-create-database: creates a database. Args: EITHER {{"database_type": "tasks"|"projects"|"skills"}} if one fits,
  OR {{"schema": "CREATE TABLE (\\"Name\\" TITLE, \\"Owner\\" RICH_TEXT, \\"Due date\\" DATE)"}} (SQL DDL syntax),
  plus {{"parent": {{"page_id": "..."}}, "title": "..."}}. Returns markdown including the new data source's
  collection://<id> — that id is what callers need afterward.

First use notion-search to find the parent page the user named. Then call notion-create-database
with a title property, an Owner property, and a Due date property, inside that page. When done,
stop calling tools and report what you created."""

_PROPERTY_MAPPING_SYSTEM_PROMPT = f"""You are mapping generic task fields onto the real property
names of a specific Notion data source.

{_TOOL_SEMANTICS}

Fetch the given data source's schema with notion-fetch. Then map:
- 'title_property' -> the property with type "title" (every data source has exactly one)
- 'owner_property' -> the best-fitting property for a person's name (types like rich_text, select, people), or null if none fits
- 'due_date_property' -> the best-fitting property of type "date", or null if none fits
- 'status_property' -> the best-fitting property for progress/status (types like status, select), or null if none fits

Use the EXACT property name strings as they appear in the schema. When done, stop calling tools
and report the mapping."""

_FIND_EXISTING_SYSTEM_PROMPT = f"""You are checking whether tasks already exist as rows in a
Notion data source, so the caller can update the existing row's progress instead of creating a
duplicate page.

{_TOOL_SEMANTICS}

For each given task title, use notion-query-data-sources (SQL mode) to search the data source's
title property for a close or exact match (e.g. "SELECT url, \\"{{title_property}}\\" FROM
\\"{{data_source_url}}\\" WHERE \\"{{title_property}}\\" LIKE '%keyword%'" — use a few
distinctive words from the title, not the whole sentence, since wording may differ slightly from
the original meeting). Only treat it as a match if the row is clearly the same task (not just
superficially similar). Use the row's "url" (or "id") field as existing_page_id. If no clear
match exists for a task, report existing_page_id as null for it — do not guess."""

_CREATE_PAGES_ENCODING_NOTES = """notion-create-pages properties are a FLAT map of
property_name -> string|number|null (not nested objects). Special encodings:
- Date properties: split into "date:{property}:start" (and optionally "date:{property}:end",
  "date:{property}:is_datetime" as 0 or 1) instead of a single key.
- Checkbox properties: use the string "__YES__" for checked, "__NO__" for unchecked.
- Properties literally named "id" or "url" (case-insensitive) must be prefixed "userDefined:".
- Every page must include its title property (the one property with type "title")."""


async def discover_data_sources(state: SessionState, task_titles: str) -> DataSourceCandidates:
    tools = mcp_tools.get_tools_by_name(state, _READ_ONLY_DISCOVERY_TOOLS)
    agent = build_notion_agent(tools)
    user_prompt = f"The extracted tasks to create: {task_titles}. List plausible target Notion databases."
    _, messages = await run_agent(agent, _DISCOVERY_SYSTEM_PROMPT, [HumanMessage(content=user_prompt)])
    messages.append(
        HumanMessage(
            content="Based on the tool results above, list every plausible candidate Notion data source "
            "you found (id — using the collection://... url from notion-fetch — name, and a page url if available)."
        )
    )
    candidates = await FptStructuredRunnable(DataSourceCandidates).ainvoke(messages)
    log.info("discover_data_sources: found %d candidate(s)", len(candidates.candidates))
    return candidates


async def create_data_source(state: SessionState, database_name: str, parent_page_name: str) -> NewDataSource:
    tools = mcp_tools.get_tools_by_name(state, _CREATE_DATABASE_TOOLS)
    agent = build_notion_agent(tools)
    user_prompt = (
        f"Create a new Notion database named '{database_name}' inside the page named "
        f"'{parent_page_name}'. If you can't find that exact page, use the closest match."
    )
    _, messages = await run_agent(agent, _CREATE_DB_SYSTEM_PROMPT, [HumanMessage(content=user_prompt)])
    messages.append(
        HumanMessage(
            content="Based on the tool results above, report the id (the collection://... data source url), "
            "name, and page url of the database you just created."
        )
    )
    result = await FptStructuredRunnable(NewDataSource).ainvoke(messages)
    log.info("create_data_source: created id=%s name=%s", result.id, result.name)
    return result


async def resolve_property_mapping(state: SessionState, data_source_id: str) -> PropertyMapping:
    """Every Notion data source has exactly one title-typed property, so a
    result with title_property=None is always wrong, not a legitimate 'no
    match' — retry once (the model occasionally omits it, observed as
    non-determinism, not a systematic error) before giving up."""
    tools = mcp_tools.get_tools_by_name(state, _READ_ONLY_DISCOVERY_TOOLS)
    user_prompt = f"Fetch the schema of the Notion data source with id '{data_source_id}' and map our task fields onto it."

    for attempt in range(2):
        agent = build_notion_agent(tools)
        _, messages = await run_agent(agent, _PROPERTY_MAPPING_SYSTEM_PROMPT, [HumanMessage(content=user_prompt)])
        messages.append(HumanMessage(content="Based on the schema you fetched, report the property mapping now."))
        mapping = await FptStructuredRunnable(PropertyMapping).ainvoke(messages)
        log.info("resolve_property_mapping: attempt=%d mapping=%s", attempt, mapping.model_dump())
        if mapping.title_property is not None:
            return mapping
        log.warning("resolve_property_mapping: got title_property=None (every data source has one), retrying")

    raise LLMResponseError(
        f"Could not determine the title property for Notion data source '{data_source_id}' "
        "after retrying. Check LOG_LEVEL=DEBUG logs for the raw schema/response."
    )


async def find_existing_tasks(
    state: SessionState, data_source_id: str, mapping: PropertyMapping, tasks: list[Task]
) -> dict[str, str]:
    """Query the data source for rows that already match each task's title, so
    create_tasks_in_notion can update the existing page's progress instead of
    creating a duplicate. Returns {task_title: existing_page_id} for tasks
    with a confident match — tasks with no match are simply absent."""
    if not tasks:
        return {}
    tools = mcp_tools.get_tools_by_name(state, _READ_ONLY_DISCOVERY_TOOLS)
    agent = build_notion_agent(tools)
    task_titles = "\n".join(f"- {t.title!r}" for t in tasks)
    user_prompt = (
        f"Data source: {data_source_id}\nTitle property name: {mapping.title_property!r}\n\n"
        f"Check whether each of these tasks already exists as a row:\n{task_titles}"
    )
    _, messages = await run_agent(agent, _FIND_EXISTING_SYSTEM_PROMPT, [HumanMessage(content=user_prompt)])
    messages.append(
        HumanMessage(content="Based on the query results above, report the match (or null) for every task title listed.")
    )
    result = await FptStructuredRunnable(ExistingTaskMatches).ainvoke(messages)
    found = {m.task_title: m.existing_page_id for m in result.matches if m.existing_page_id}
    log.info("find_existing_tasks: %d/%d task(s) already exist", len(found), len(tasks))
    return found


_UUID_HEX_RE = re.compile(r"([0-9a-f]{8})-?([0-9a-f]{4})-?([0-9a-f]{4})-?([0-9a-f]{4})-?([0-9a-f]{12})", re.IGNORECASE)


def _extract_uuid(value: str) -> str:
    """Notion ids surface in many shapes depending on which tool/agent surfaced
    them — collection://<uuid>, full https://app.notion.com/<32-hex-no-dashes>
    URLs, dashed or undashed UUIDs. But notion-create-pages/notion-update-page
    both validate their id args as a bare dashed UUID and reject any prefix or
    URL wrapper (confirmed via live 400 responses for both tools). Extract the
    32 hex chars from wherever they appear and re-dash them, rather than
    special-casing each possible input shape."""
    match = _UUID_HEX_RE.search(value)
    if not match:
        return value
    return "-".join(match.groups())


def _bare_data_source_uuid(data_source_id: str) -> str:
    """notion-fetch/discovery surface data source ids as collection://<uuid>
    (needed so the agent can recognize/re-quote them from markdown), but the
    real notion-create-pages API validates parent.data_source_id as a bare
    UUID and rejects the collection:// prefix — confirmed via a live 400
    response ('data_source_id should be a valid uuid, instead was
    "collection://...").)"""
    return _extract_uuid(data_source_id)


def _task_body(task: Task, mapping: PropertyMapping) -> str:
    lines = [task.description] if task.description else []
    if task.owner and mapping.owner_property is None:
        lines.append(f"Owner: {task.owner}")
    if task.due_date and mapping.due_date_property is None:
        lines.append(f"Due date: {task.due_date}")
    if task.status and mapping.status_property is None:
        lines.append(f"Status: {task.status}")
    if task.dependencies:
        lines.append("Depends on: " + ", ".join(task.dependencies))
    if task.source_excerpt:
        lines.append(f'Source: "{task.source_excerpt}"')
    return "\n".join(lines)


def _property_mapping_prompt(mapping: PropertyMapping) -> str:
    return (
        f"title_property->{mapping.title_property!r}, owner_property->{mapping.owner_property!r}, "
        f"due_date_property->{mapping.due_date_property!r}, status_property->{mapping.status_property!r}"
    )


_STATUS_ENUM_NOTE = """If the status/select property has a fixed set of options (visible in its schema
as an "options" list), you MUST use notion-fetch first to read those exact option names, then map the
task's free-text status onto the closest matching option — never invent a value outside that list.
If the property type has no fixed options (e.g. rich_text), use the task's status text as-is."""


async def _create_new_tasks(
    state: SessionState, bare_data_source_id: str, mapping: PropertyMapping, tasks: list[Task]
) -> list[CreatePageOutcome]:
    """Batch-creates brand-new pages for tasks with no existing match. Uses a
    read-only agent (with notion-fetch) to resolve any fixed-option status
    value before building the create-pages args, then performs the actual
    write as a single direct tool call — not inside the agent loop."""
    if not tasks:
        return []
    create_pages_tools = mcp_tools.get_tools_by_name(state, {_CREATE_PAGES_TOOL})
    if not create_pages_tools:
        raise RuntimeError(f"No connected Notion MCP tool named '{_CREATE_PAGES_TOOL}'")
    create_pages_tool = create_pages_tools[0]

    tasks_summary = "\n".join(
        f"- title={t.title!r} owner={t.owner!r} due_date={t.due_date!r} status={t.status!r} "
        f"body={_task_body(t, mapping)!r}"
        for t in tasks
    )
    prompt = (
        f"{_CREATE_PAGES_ENCODING_NOTES}\n\n{_STATUS_ENUM_NOTE}\n\n"
        f"Property mapping: {_property_mapping_prompt(mapping)}\n"
        f"Parent data source id: {bare_data_source_id} (use notion-fetch with 'collection://{bare_data_source_id}' "
        "if you need to check the schema)\n\n"
        f"Build the notion-create-pages tool call arguments (parent + pages list) for these tasks:\n{tasks_summary}\n\n"
        "Use the given data_source_id (bare UUID, no prefix) as the parent's data_source_id. STRICT RULE: the ONLY "
        "properties you may set are the exact property names given in the mapping above — do not invent, guess, "
        "or add any other property name under any circumstances, even if a task has an owner/due_date/status "
        "value. If a mapping property is None, do NOT set any similarly-purposed property (that info is already "
        "included in the task's body text instead). Set the title_property to the task title. Use the task's "
        "body text as \"content\"."
    )
    tools = mcp_tools.get_tools_by_name(state, {"notion-fetch"})
    agent = build_notion_agent(tools)
    _, messages = await run_agent(agent, "You are preparing Notion page creation arguments.", [HumanMessage(content=prompt)])
    messages.append(HumanMessage(content="Now report the final notion-create-pages arguments."))
    args = await FptStructuredRunnable(NotionCreatePagesArgs).ainvoke(messages)

    raw, ok = await mcp_tools.call_tool_directly(create_pages_tool, args.model_dump())
    log.info("create_tasks_in_notion: create-pages call ok=%s (%d task(s))", ok, len(tasks))
    return [CreatePageOutcome(task_title=t.title, ok=ok, action="created", detail=raw if not ok else "created") for t in tasks]


async def _update_existing_task(
    state: SessionState,
    bare_data_source_id: str,
    mapping: PropertyMapping,
    task: Task,
    page_id: str,
) -> CreatePageOutcome:
    """Updates just the progress/status of an already-existing page — does not
    touch title/owner/due_date, since those aren't what changed. Uses a
    read-only agent (with notion-fetch) to resolve any fixed-option status
    value before the single direct notion-update-page write call."""
    update_tools = mcp_tools.get_tools_by_name(state, {_UPDATE_PAGE_TOOL})
    if not update_tools:
        raise RuntimeError(f"No connected Notion MCP tool named '{_UPDATE_PAGE_TOOL}'")
    update_tool = update_tools[0]

    if mapping.status_property is None or task.status is None:
        return CreatePageOutcome(
            task_title=task.title,
            ok=True,
            action="unchanged",
            detail="already exists; no status property/value to update",
        )

    bare_page_id = _extract_uuid(page_id)
    prompt = (
        f"{_CREATE_PAGES_ENCODING_NOTES}\n\n{_STATUS_ENUM_NOTE}\n\n"
        f"Page id to update (bare UUID, do not add any prefix or URL wrapper to it): {bare_page_id!r}\n"
        f"Status property name: {mapping.status_property!r}\n"
        f"Data source id (fetch 'collection://{bare_data_source_id}' with notion-fetch to check the schema/options): "
        f"{bare_data_source_id}\n"
        f"New status value to map onto the real property: {task.status!r}\n\n"
        "Build the notion-update-page tool call arguments: command must be \"update_properties\", and "
        "properties must contain ONLY the status property (do not include title, owner, due date, or any "
        "other property — this call must only change the status)."
    )
    tools = mcp_tools.get_tools_by_name(state, {"notion-fetch"})
    agent = build_notion_agent(tools)
    _, messages = await run_agent(agent, "You are preparing a Notion page update.", [HumanMessage(content=prompt)])
    messages.append(HumanMessage(content="Now report the final notion-update-page arguments."))
    args = await FptStructuredRunnable(NotionUpdatePageArgs).ainvoke(messages)

    raw, ok = await mcp_tools.call_tool_directly(update_tool, args.model_dump())
    log.info("create_tasks_in_notion: update-page call ok=%s for task=%r", ok, task.title)
    return CreatePageOutcome(
        task_title=task.title, ok=ok, action="updated", detail=raw if not ok else f"status updated to {task.status!r}"
    )


async def create_tasks_in_notion(
    state: SessionState,
    data_source_id: str,
    mapping: PropertyMapping,
    tasks: list[Task],
    existing_page_ids: dict[str, str] | None = None,
) -> list[CreatePageOutcome]:
    """The single Notion write call site in the whole app. Tasks with a known
    existing_page_ids match get their progress/status updated in place;
    everything else is batch-created as new pages. Not an agent loop for
    either path — one structured-generation call builds the exact tool
    arguments, then the tool is invoked directly, only after app.py's
    confirmation gate."""
    existing_page_ids = existing_page_ids or {}
    bare_data_source_id = _bare_data_source_uuid(data_source_id)

    new_tasks = [t for t in tasks if t.title not in existing_page_ids]
    existing_tasks = [t for t in tasks if t.title in existing_page_ids]

    results: list[CreatePageOutcome] = []
    for task in existing_tasks:
        results.append(
            await _update_existing_task(state, bare_data_source_id, mapping, task, existing_page_ids[task.title])
        )
    results.extend(await _create_new_tasks(state, bare_data_source_id, mapping, new_tasks))
    return results
