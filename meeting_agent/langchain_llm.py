import json
import re

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from openai import APIError, AsyncOpenAI
from pydantic import BaseModel

from . import config
from .logging_config import get_logger

log = get_logger("langchain_llm")

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# ChatOpenAI (langchain-openai) explicitly does not preserve non-standard
# response fields like `reasoning_content` when base_url points at a
# non-OpenAI provider (see its own docstring warning) — verified live against
# the FPT Cloud endpoint: the raw OpenAI SDK exposes
# `message.model_extra == {"reasoning_content": "{}"}` on an empty-content
# response, but ChatOpenAI's parsed AIMessage.additional_kwargs drops it
# entirely. So the structured-output attempt (the one that needs to detect
# this quirk) goes through the raw AsyncOpenAI client, not ChatOpenAI.
_raw_client = AsyncOpenAI(api_key=config.FPT_API_KEY, base_url=config.FPT_BASE_URL)


class LLMResponseError(RuntimeError):
    """Raised when the LLM call fails or its response can't be used (API error, empty content, refusal, etc.)."""


def build_chat_model(temperature: float = 0.2) -> ChatOpenAI:
    """Plain (non-structured) chat model for agent/tool-calling use, where the
    response_format empty-content quirk does not apply."""
    return ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.FPT_API_KEY,
        base_url=config.FPT_BASE_URL,
        temperature=temperature,
        max_retries=0,
    )


def _to_messages(value) -> list[BaseMessage]:
    if isinstance(value, PromptValue):
        return value.to_messages()
    if isinstance(value, list):
        return value
    raise TypeError(f"FptStructuredRunnable expects a list[BaseMessage] or PromptValue, got {type(value)}")


def _lc_to_openai_messages(messages: list[BaseMessage]) -> list[dict]:
    role_map = {"system": "system", "human": "user", "ai": "assistant", "tool": "tool"}
    out = []
    for m in messages:
        role = role_map.get(m.type, m.type)
        entry = {"role": role, "content": m.content or ""}
        if m.type == "tool":
            entry["tool_call_id"] = m.tool_call_id
        if m.type == "ai" and getattr(m, "tool_calls", None):
            entry["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": _json_dumps(tc["args"])},
                }
                for tc in m.tool_calls
            ]
        out.append(entry)
    return out


def _json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _extract_json_text(content: str) -> str:
    fence_match = _JSON_FENCE_RE.search(content)
    return fence_match.group(1) if fence_match else content


def _non_empty_content(choice) -> str | None:
    content = choice.message.content
    if content and content.strip() not in ("", "{}"):
        return content
    reasoning_content = getattr(choice.message, "reasoning_content", None) or (
        choice.message.model_extra.get("reasoning_content") if choice.message.model_extra else None
    )
    if reasoning_content and reasoning_content.strip() not in ("", "{}"):
        return reasoning_content
    return None


class FptStructuredRunnable(Runnable):
    """Requests JSON matching a Pydantic model from the FPT Cloud endpoint.

    Primary path is a plain prompt-based JSON request (no response_format) via
    ChatOpenAI. This is deliberate, not a fallback-of-last-resort: repeated
    testing showed that binding response_format=json_schema on this
    endpoint/model doesn't just risk empty content (the original quirk) — on
    longer/more complex inputs it sometimes returns syntactically VALID but
    WRONG json (e.g. {"tasks": []} when the transcript clearly has 5 tasks),
    a reasoning-quality regression, not a technical failure _non_empty_content
    can detect. Plain-prompt mode consistently produced correct results across
    many repeated runs of the same inputs, so it's the default.

    response_format=json_schema (via the raw OpenAI SDK, so the
    `reasoning_content` provider quirk field stays visible — ChatOpenAI drops
    it) is used only as a fallback, if the plain-prompt attempt returns
    unparseable/empty content. Returns a validated instance of `schema`, not a
    raw string.
    """

    def __init__(self, schema: type[BaseModel], temperature: float = 0.2):
        self._schema = schema
        self._schema_dict = schema.model_json_schema()
        self._temperature = temperature
        self._plain_model = build_chat_model(temperature=temperature)

    async def ainvoke(self, input, config=None, **kwargs) -> BaseModel:
        messages = _to_messages(input)
        plain_messages = [
            *messages,
            HumanMessage(
                content=(
                    "Return ONLY a JSON object matching this schema, no markdown fences, no explanation:\n"
                    f"{json.dumps(self._schema_dict, ensure_ascii=False)}"
                )
            ),
        ]
        ai_msg = await self._plain_model.ainvoke(plain_messages)
        log.debug("FptStructuredRunnable: plain-prompt raw content for schema=%s: %r", self._schema.__name__, ai_msg.content)
        content = ai_msg.content if ai_msg.content and ai_msg.content.strip() not in ("", "{}") else None
        if content:
            try:
                return self._parse(_extract_json_text(content))
            except Exception:
                log.warning(
                    "FptStructuredRunnable: plain-prompt content for schema=%s didn't parse (raw=%r), "
                    "falling back to response_format=json_schema",
                    self._schema.__name__,
                    content,
                )
        else:
            log.warning(
                "FptStructuredRunnable: empty plain-prompt content for schema=%s, "
                "falling back to response_format=json_schema",
                self._schema.__name__,
            )

        openai_messages = _lc_to_openai_messages(messages)
        structured_content = await self._try_structured(openai_messages)
        if not structured_content:
            log.error(
                "FptStructuredRunnable: empty content for schema=%s from both plain-prompt and "
                "response_format attempts",
                self._schema.__name__,
            )
            raise LLMResponseError(
                f"LLM returned no usable content for schema '{self._schema.__name__}' via either "
                "plain-prompt or structured output. Check LOG_LEVEL=DEBUG logs for the raw response."
            )
        return self._parse(structured_content)

    def invoke(self, input, config=None, **kwargs) -> BaseModel:
        import asyncio

        return asyncio.run(self.ainvoke(input, config=config, **kwargs))

    async def _try_structured(self, openai_messages: list[dict]) -> str | None:
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": self._schema.__name__, "schema": self._schema_dict, "strict": True},
        }
        try:
            resp = await _raw_client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=openai_messages,
                response_format=response_format,
                temperature=self._temperature,
            )
        except APIError as e:
            log.error("FptStructuredRunnable: API call failed: %s", e)
            raise LLMResponseError(f"LLM API call failed: {e}") from e

        choice = resp.choices[0]
        log.debug(
            "FptStructuredRunnable: structured attempt finish_reason=%s content_len=%s",
            choice.finish_reason,
            len(choice.message.content) if choice.message.content else 0,
        )
        content = _non_empty_content(choice)
        return _extract_json_text(content) if content else None

    def _parse(self, content: str) -> BaseModel:
        return self._schema.model_validate_json(content)
