from pydantic import BaseModel, Field


class DataSourceCandidate(BaseModel):
    id: str = Field(
        description="Notion data source id usable as a parent for notion-create-pages "
        "(e.g. the collection://<id> value found in a notion-fetch result), "
        "or a page id/URL if no specific data source was identified"
    )
    name: str
    url: str | None = None


class DataSourceCandidates(BaseModel):
    candidates: list[DataSourceCandidate] = Field(default_factory=list)


class NewDataSource(BaseModel):
    id: str = Field(description="The data source id (collection://<id>) of the newly created database")
    name: str
    url: str | None = None


class PropertyMapping(BaseModel):
    """Real Notion property NAMES (as they must appear as keys when calling
    notion-create-pages), resolved by an LLM reading the data source's actual
    markdown schema — not type-annotated metadata, just the name strings.

    Field names deliberately avoid "title" on its own: Pydantic's generated
    JSON Schema always has a top-level "title" keyword set to the class name
    (e.g. "PropertyMapping"), and a field also named "title" is easily
    confused with that keyword by an LLM reading the raw schema in
    plain-prompt mode — this was observed to cause the model to literally
    emit the class name ("PropertyMapping") as the field value instead of the
    real Notion property name."""

    title_property: str | None = Field(
        default=None, description="Exact property name to use as the page title"
    )
    owner_property: str | None = Field(
        default=None, description="Exact property name for task owner, or null if none suitable"
    )
    due_date_property: str | None = Field(
        default=None, description="Exact property name for due date, or null if none suitable"
    )
    status_property: str | None = Field(
        default=None,
        description="Exact property name for progress/status (types like status, select), or null if none suitable",
    )


class NotionPageSpec(BaseModel):
    """A single page within notion-create-pages' `pages` array. Properties is
    its own nested object — deliberately NOT a flat `dict` for the whole page,
    so the JSON Schema forces the model to nest property key/value pairs under
    `properties` instead of placing them as siblings of `properties`/`content`
    (observed failure: the model emitted {"Name": ..., "Status": ..., "content":
    ...} directly under a page item, which notion-create-pages rejected with
    "Unrecognized keys" since it has no top-level "Name"/"Status" keys)."""

    properties: dict = Field(description="Flat name->value map of ONLY the properties named in the mapping")
    content: str = Field(default="", description="Markdown body text for the page")


class NotionCreatePagesArgs(BaseModel):
    """Mirrors the real notion-create-pages tool input shape."""

    parent: dict[str, str] = Field(description='e.g. {"data_source_id": "..."} or {"type": "data_source_id", "data_source_id": "..."}')
    pages: list[NotionPageSpec]


class NotionUpdatePageArgs(BaseModel):
    """Mirrors the real notion-update-page tool input shape for a single
    update_properties call."""

    page_id: str
    command: str = Field(default="update_properties")
    properties: dict = Field(description="Flat name->value map of ONLY the properties to change")


class ExistingTaskMatch(BaseModel):
    task_title: str = Field(description="The extracted task's title, exactly as given")
    existing_page_id: str | None = Field(
        default=None, description="Page id of the matching existing Notion row, or null if no match was found"
    )


class ExistingTaskMatches(BaseModel):
    matches: list[ExistingTaskMatch] = Field(default_factory=list)


class CreatePageOutcome(BaseModel):
    task_title: str
    ok: bool
    action: str = Field(default="created", description="'created' or 'updated'")
    detail: str
