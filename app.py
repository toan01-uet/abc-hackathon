from meeting_agent.logging_config import setup_logging

# configure logging first, then import the agent module which registers Chainlit handlers
setup_logging()
from pathlib import Path

# Ensure the `.files` directory exists so Chainlit can create per-session
# subdirectories without raising FileNotFoundError on Windows when parents
# don't yet exist. Use parents=True to create intermediate directories.
try:
	files_dir = Path(__file__).parent / ".files"
	files_dir.mkdir(parents=True, exist_ok=True)
except Exception:
	# Best-effort; if this fails the Chainlit server may still create the
	# directory when handling uploads, or we'll surface an error in logs.
	pass

import meeting_agent.agent_graph  # registers Chainlit handlers and graph
