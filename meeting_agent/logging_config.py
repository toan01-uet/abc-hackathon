import logging
import os

_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt=_LOG_DATEFMT)

    log_file = os.environ.get("LOG_FILE")
    if log_file:
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
        logging.getLogger("meeting_agent").addHandler(handler)

    logging.getLogger("meeting_agent").setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"meeting_agent.{name}")
