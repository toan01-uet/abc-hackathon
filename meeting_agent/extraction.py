import json

from . import llm
from .models import TaskList

_SCHEMA = TaskList.model_json_schema()

_SYSTEM_PROMPT = """You are the extraction module of a Meeting Tasks Agent.
Read a meeting transcript (which may be in English or Vietnamese) and extract concrete action items as structured tasks.

For each task, identify:
- title: short, action-oriented (e.g. "Send logo assets to John")
- description: one or two sentences of context
- owner: the person responsible, if named in the transcript, else null
- due_date: an ISO 8601 date (YYYY-MM-DD) only if a specific date is stated or unambiguously derivable; otherwise null
- dependencies: titles of other extracted tasks that must finish first, if the transcript implies an ordering
- source_excerpt: a short verbatim quote from the transcript backing this task

Do not invent facts that aren't stated or clearly implied by the transcript. If no action items are present, return an empty task list."""


async def extract_tasks(transcript: str) -> TaskList:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ]
    content = await llm.structured_completion(messages=messages, schema_name="task_list", schema=_SCHEMA)
    return TaskList.model_validate_json(content)


async def revise_tasks(tasks: TaskList, feedback: str) -> TaskList:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Here is the current extracted task list as JSON:\n"
                f"{json.dumps(tasks.model_dump(), ensure_ascii=False)}\n\n"
                "The user has this feedback/correction request:\n"
                f"{feedback}\n\n"
                "Return the full, revised task list applying this feedback."
            ),
        },
    ]
    content = await llm.structured_completion(messages=messages, schema_name="task_list", schema=_SCHEMA)
    return TaskList.model_validate_json(content)
