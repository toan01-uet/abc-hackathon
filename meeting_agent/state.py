from dataclasses import dataclass, field
from typing import Any, Literal

import chainlit as cl
from mcp import ClientSession

from .models import Task

Stage = Literal["collecting_transcript", "reviewing_tasks", "confirming", "creating", "done"]

_STATE_KEY = "meeting_agent_state"


@dataclass
class SessionState:
    stage: Stage = "collecting_transcript"
    transcript: str | None = None
    tasks: list[Task] = field(default_factory=list)
    mcp_tool_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    mcp_clients: dict[str, ClientSession] = field(default_factory=dict)
    notion_data_source_id: str | None = None
    notion_data_source_name: str | None = None
    property_mapping: dict | None = None
    write_confirmed: bool = False


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
