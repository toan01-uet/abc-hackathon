"""Quick, Chainlit-free sanity check for the extraction module.

Usage:
    uv run python scripts/smoke_extract.py samples/transcript_en.txt
    uv run python scripts/smoke_extract.py samples/transcript_vi.txt
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meeting_agent import extraction  # noqa: E402


async def main(path: str):
    transcript = Path(path).read_text(encoding="utf-8")
    tasks = await extraction.extract_tasks(transcript)
    print(tasks.model_dump_json(indent=2, exclude_none=False))


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "samples/transcript_en.txt"
    asyncio.run(main(target))
