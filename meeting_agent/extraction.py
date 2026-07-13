import json

from langchain_core.prompts import ChatPromptTemplate

from .langchain_llm import FptStructuredRunnable
from .logging_config import get_logger
from .models import TaskList

log = get_logger("extraction")

_SYSTEM_PROMPT = """You are the extraction module of a Meeting Tasks Agent.
Read a meeting transcript (which may be in English or Vietnamese) and extract concrete action items as structured tasks.

For each task, identify:
- title: short, action-oriented (e.g. "Send logo assets to John")
- description: one or two sentences of context
- owner: the person responsible, if named in the transcript, else null
- due_date: an ISO 8601 date (YYYY-MM-DD) only if a specific date is stated or unambiguously derivable; otherwise null
- dependencies: titles of other extracted tasks that must finish first, if the transcript implies an ordering
- source_excerpt: a short verbatim quote from the transcript backing this task
- status: the task's progress state, inferred from how it's discussed (e.g. someone says they already finished it,
  are currently working on it, or haven't started yet). Use a short phrase in the transcript's own language
  (e.g. "Done", "In progress", "Not started", "Đã xong", "Đang làm"). Use null if the transcript gives no hint
  either way.

Do not invent facts that aren't stated or clearly implied by the transcript. If no action items are present, return an empty task list."""

_extract_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", "{transcript}"),
    ]
)

_revise_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        (
            "human",
            "Here is the current extracted task list as JSON:\n{current_tasks_json}\n\n"
            "The user has this feedback/correction request:\n{feedback}\n\n"
            "Return the full, revised task list applying this feedback.",
        ),
    ]
)

_extract_chain = _extract_prompt | FptStructuredRunnable(TaskList)
_revise_chain = _revise_prompt | FptStructuredRunnable(TaskList)


async def extract_tasks(transcript: str) -> TaskList:
    log.info("extract_tasks: transcript_len=%d", len(transcript))
    task_list = await _extract_chain.ainvoke({"transcript": transcript})
    log.info("extract_tasks: extracted %d task(s)", len(task_list.tasks))
    return task_list


async def revise_tasks(tasks: TaskList, feedback: str) -> TaskList:
    log.info("revise_tasks: current_tasks=%d feedback=%r", len(tasks.tasks), feedback)
    task_list = await _revise_chain.ainvoke(
        {
            "current_tasks_json": json.dumps(tasks.model_dump(), ensure_ascii=False),
            "feedback": feedback,
        }
    )
    log.info("revise_tasks: revised to %d task(s)", len(task_list.tasks))
    return task_list
