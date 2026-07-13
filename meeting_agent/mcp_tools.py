from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession

from .logging_config import get_logger
from .state import SessionState

log = get_logger("mcp_tools")


async def register_mcp_connection(state: SessionState, name: str, session: ClientSession) -> int:
    tools = await load_mcp_tools(session, server_name=name, handle_tool_errors=True)
    state.mcp_tool_cache[name] = {t.name: t for t in tools}
    state.mcp_clients[name] = session
    log.info("MCP connected: name=%s tools=%s", name, [t.name for t in tools])
    return len(tools)


def unregister_mcp_connection(state: SessionState, name: str) -> None:
    state.mcp_tool_cache.pop(name, None)
    state.mcp_clients.pop(name, None)
    log.info("MCP disconnected: name=%s", name)


def get_all_tools(state: SessionState) -> list[BaseTool]:
    """Aggregate every connected MCP server's tools into one combined toolset."""
    return [tool for tools in state.mcp_tool_cache.values() for tool in tools.values()]


def get_tools_by_name(state: SessionState, names: set[str]) -> list[BaseTool]:
    """Filter the aggregated toolset down to an explicit allowlist of tool names.
    Used to restrict an agent to read-only tools, or to hand a single write
    tool to a non-agentic call site."""
    tools = [t for t in get_all_tools(state) if t.name in names]
    missing = names - {t.name for t in tools}
    if missing:
        log.warning("get_tools_by_name: requested tools not found on any connected MCP server: %s", missing)
    return tools


async def call_tool_directly(tool: BaseTool, arguments: dict) -> tuple[str, bool]:
    """Invoke a single tool outside of any agent loop, e.g. the one Notion
    write call site. Uses a synthetic ToolCall so langchain-mcp-adapters'
    error handling produces a real ToolMessage with a `.status`, rather than
    the bare content list _format_output returns when no tool_call_id is given
    (which carries no success/failure signal at all)."""
    result = await tool.ainvoke(
        {"type": "tool_call", "name": tool.name, "args": arguments, "id": "manual-call"}
    )
    ok = getattr(result, "status", "success") != "error"
    content = result.content if hasattr(result, "content") else str(result)
    if not ok:
        log.error("call_tool_directly: tool=%s returned error: %s", tool.name, content)
    else:
        log.debug("call_tool_directly: tool=%s ok, content_len=%d", tool.name, len(str(content)))
    return str(content), ok
