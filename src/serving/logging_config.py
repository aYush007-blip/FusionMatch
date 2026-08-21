"""Structured JSON Logging Configuration with Loguru."""

import sys
import json
from loguru import logger


def setup_logger(log_level: str = "INFO", json_format: bool = False) -> None:
    """Configures Loguru logger with structured formatting and standard stream routing."""
    logger.remove()

    if json_format or (not sys.stdout.isatty()):
        def json_sink(message):
            record = message.record
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
            sys.stdout.write(json.dumps(subset) + "\n")
            sys.stdout.flush()

        logger.add(
            json_sink,
            level=log_level.upper(),
        )
    else:
        logger.add(
            sys.stdout,
            level=log_level.upper(),
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True,
        )
