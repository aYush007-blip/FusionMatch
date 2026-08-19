"""Structured JSON Logging Configuration with Loguru."""

import sys
import json
from loguru import logger


def setup_logger(log_level: str = "INFO") -> None:
    """Configures Loguru logger with structured formatting and standard stream routing."""
    logger.remove()

    def serialize(record):
        subset = {
            "timestamp": record["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "level": record["level"].name,
            "message": record["message"],
            "module": record["name"],
            "function": record["function"],
            "line": record["line"],
        }
        if record["extra"]:
            subset["extra"] = record["extra"]
        return json.dumps(subset) + "\n"

    logger.add(
        sys.stdout,
        level=log_level.upper(),
        format="{message}",
        colorize=False,
    )
