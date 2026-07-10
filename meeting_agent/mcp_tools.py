import json

from mcp import ClientSession

from . import llm
from .state import SessionState


def mcp_tool_to_openai_schema(tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


async def register_mcp_connection(state: SessionState, name: str, session: ClientSession) -> int:
    result = await session.list_tools()
    state.mcp_tool_cache[name] = {t.name: t for t in result.tools}
    state.mcp_clients[name] = session
    return len(result.tools)


def unregister_mcp_connection(state: SessionState, name: str) -> None:
    state.mcp_tool_cache.pop(name, None)
    state.mcp_clients.pop(name, None)


def get_openai_tools(state: SessionState) -> list[dict]:
    return [
        mcp_tool_to_openai_schema(tool)
        for tools in state.mcp_tool_cache.values()
        for tool in tools.values()
    ]


def _find_owner(state: SessionState, tool_name: str) -> tuple[str, ClientSession] | None:
    for conn_name, tools in state.mcp_tool_cache.items():
        if tool_name in tools:
            return conn_name, state.mcp_clients[conn_name]
    return None


async def dispatch_tool_call(state: SessionState, tool_name: str, arguments: dict) -> str:
    owner = _find_owner(state, tool_name)
    if owner is None:
        return f"ERROR: tool '{tool_name}' not found on any connected MCP server"
    _, client = owner
    result = await client.call_tool(tool_name, arguments)
    text = "\n".join(block.text for block in result.content if hasattr(block, "text"))
    return f"ERROR: {text}" if result.isError else text


async def run_tool_calling_loop(
    state: SessionState,
    system_prompt: str,
    messages: list[dict],
    max_turns: int = 8,
) -> tuple[str, list[dict]]:
    """Runs a manual agentic loop: the LLM sees the discovered MCP tools, may emit
    tool_calls, we dispatch each via the owning MCP session and feed results back,
    until the LLM returns plain text. Returns (final_text, updated_messages)."""
    tools = get_openai_tools(state)
    full_messages = [{"role": "system", "content": system_prompt}, *messages]
    for _ in range(max_turns):
        resp = await llm.chat_completion(messages=full_messages, tools=tools, tool_choice="auto")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or "", full_messages
        full_messages.append(msg.model_dump(exclude_none=True))
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            content = await dispatch_tool_call(state, tc.function.name, args)
            full_messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
    return "Stopped after max tool-call turns without a final answer.", full_messages
