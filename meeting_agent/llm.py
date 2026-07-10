from openai import AsyncOpenAI

from . import config

client = AsyncOpenAI(api_key=config.FPT_API_KEY, base_url=config.FPT_BASE_URL)


async def chat_completion(*, messages, tools=None, tool_choice=None, response_format=None, temperature=0.2):
    kwargs = {"model": config.LLM_MODEL, "messages": messages, "temperature": temperature}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice or "auto"
    if response_format:
        kwargs["response_format"] = response_format
    return await client.chat.completions.create(**kwargs)


async def structured_completion(*, messages, schema_name: str, schema: dict, temperature=0.2) -> str:
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": schema_name, "schema": schema, "strict": True},
    }
    resp = await chat_completion(messages=messages, response_format=response_format, temperature=temperature)
    return resp.choices[0].message.content
