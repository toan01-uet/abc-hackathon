import os

from dotenv import load_dotenv

load_dotenv()

FPT_API_KEY = os.environ["FPT_API_KEY"]
FPT_BASE_URL = os.environ.get("FPT_BASE_URL", "https://mkp-api.fptcloud.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "DeepSeek-V4-Flash")
