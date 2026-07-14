from dataclasses import dataclass, field
from typing import Literal

import chainlit as cl
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from mcp import ClientSession

from .models import Task
from .notion_models import PropertyMapping

Stage = Literal["collecting_transcript", "reviewing_tasks", "confirming", "creating", "done"]

_STATE_KEY = "meeting_agent_state"


@dataclass
class SessionState:
    stage: Stage = "collecting_transcript"
    transcript: str | None = None
    tasks: list[Task] = field(default_factory=list)
    mcp_tool_cache: dict[str, dict[str, BaseTool]] = field(default_factory=dict)
    mcp_clients: dict[str, ClientSession] = field(default_factory=dict)
    notion_data_source_id: str | None = None
    notion_data_source_name: str | None = None
    property_mapping: PropertyMapping | None = None
    existing_page_ids: dict[str, str] = field(default_factory=dict)
    write_confirmed: bool = False
    # Graph runtime helpers
    awaiting_input: bool = False
    graph_prompt: str | None = None
    graph_candidates: list | None = None
    graph_thread_id: str | None = None
    graph_placeholder_id: str | None = None
    # Optional override route set by the LLM router; consumed by the next router invocation
    graph_route_override: str | None = None
    # Accumulated conversation history (HumanMessage / AIMessage objects)
    chat_messages: list = field(default_factory=list)


def get_state() -> SessionState:
    state = cl.user_session.get(_STATE_KEY)
    if state is None:
        state = SessionState()
        cl.user_session.set(_STATE_KEY, state)
    return state


def reset_state() -> SessionState:
    state = SessionState()
    cl.user_session.set(_STATE_KEY, state)
    return state
