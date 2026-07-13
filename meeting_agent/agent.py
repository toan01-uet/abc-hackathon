from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from .langchain_llm import build_chat_model
from .logging_config import get_logger

log = get_logger("agent")


def build_notion_agent(tools: list[BaseTool]):
    """A read-only-or-narrowly-scoped ReAct agent over an explicit toolset.
    The toolset passed in is the only safety boundary — never pass a write
    tool like notion-create-pages unless the caller has already obtained
    explicit user intent for that specific write."""
    return create_react_agent(model=build_chat_model(), tools=tools)


async def run_agent(
    agent, system_prompt: str, user_messages: list[BaseMessage], max_turns: int = 8
) -> tuple[str, list[BaseMessage]]:
    """Drop-in replacement for the old manual run_tool_calling_loop: runs the
    agent to completion (or until max_turns is exhausted) and returns
    (final_text, full_message_history) for logging/debugging and for
    follow-up structured-output calls over the accumulated messages."""
    log.info("run_agent: start, max_turns=%d", max_turns)
    result = await agent.ainvoke(
        {"messages": [SystemMessage(content=system_prompt), *user_messages]},
        config={"recursion_limit": 2 * max_turns + 1},
    )
    messages = result["messages"]
    last = messages[-1] if messages else None
    final_text = last.content if last is not None and not getattr(last, "tool_calls", None) else ""
    log.info("run_agent: finished, message_count=%d", len(messages))
    return final_text, messages
