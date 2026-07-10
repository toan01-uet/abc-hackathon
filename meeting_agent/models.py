from pydantic import BaseModel, Field


class Task(BaseModel):
    title: str
    description: str = ""
    owner: str | None = None
    due_date: str | None = Field(default=None, description="ISO 8601 date (YYYY-MM-DD) if mentioned, else null")
    dependencies: list[str] = Field(default_factory=list, description="Titles of other tasks this one depends on")
    source_excerpt: str = Field(default="", description="Short quote from the transcript this task was derived from")


class TaskList(BaseModel):
    tasks: list[Task] = Field(default_factory=list)
