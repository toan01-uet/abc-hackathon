from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langchain.agents import create_agent

from .langchain_llm import LLMResponseError, build_chat_model
from .logging_config import get_logger

log = get_logger("agent")


def build_notion_agent(tools: list[BaseTool]):
    """A read-only-or-narrowly-scoped ReAct agent over an explicit toolset.
    The toolset passed in is the only safety boundary — never pass a write
    tool like notion-create-pages unless the caller has already obtained
    explicit user intent for that specific write."""
    return create_agent(model=build_chat_model(), tools=tools)


def _log_agent_update(node_name: str, node_data: dict) -> None:
    for msg in node_data.get("messages", []):
        if node_name == "agent":
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    log.info("agent: calling tool=%s args=%s", tc["name"], tc["args"])
            reasoning = (msg.content or "").strip()
            if reasoning:
                log.info("agent: reasoning: %s", reasoning)
        elif node_name == "tools":
            content = str(msg.content)
            status = getattr(msg, "status", "success")
            preview = content if len(content) <= 500 else content[:500] + "…"
            log.info("agent: tool=%s status=%s result=%s", getattr(msg, "name", "?"), status, preview)


async def run_agent(
    agent, system_prompt: str, user_messages: list[BaseMessage], max_turns: int = 100
) -> tuple[str, list[BaseMessage]]:
    """Drop-in replacement for the old manual run_tool_calling_loop: runs the
    agent to completion (or until max_turns is exhausted) and returns
    (final_text, full_message_history) for logging/debugging and for
    follow-up structured-output calls over the accumulated messages.

    Streams via astream(stream_mode="updates") instead of a single ainvoke()
    so each reasoning step and tool call is logged as it happens (visible with
    LOG_LEVEL=INFO), not just a start/finished pair.

    Unlike the old hand-rolled loop, LangGraph's recursion limit is a hard
    stop: streaming *raises* GraphRecursionError instead of returning
    whatever partial progress it made, so that's converted here into a
    LLMResponseError callers already know how to handle (surfaced as a
    friendly chat message instead of a raw traceback)."""
    log.info("run_agent: start, max_turns=%d", max_turns)
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt), *user_messages]
    try:
        async for chunk in agent.astream(
            {"messages": messages},
            config={"recursion_limit": 2 * max_turns + 1},
            stream_mode="updates",
        ):
            for node_name, node_data in chunk.items():
                _log_agent_update(node_name, node_data)
                messages.extend(node_data.get("messages", []))
    except GraphRecursionError as e:
        log.error("run_agent: hit recursion limit (max_turns=%d) without finishing", max_turns)
        raise LLMResponseError(
            f"The agent couldn't finish within {max_turns} tool-calling turns. This can happen if "
            "the model keeps calling tools without settling on an answer — try again, or simplify "
            "the request."
        ) from e
    last = messages[-1] if messages else None
    final_text = last.content if last is not None and not getattr(last, "tool_calls", None) else ""
    log.info("run_agent: finished, message_count=%d", len(messages))
    return final_text, messages
