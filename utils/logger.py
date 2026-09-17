"""Logging setup for the Byte Latent Transformer (BLT) project.

Provides unified formatting for console and file logging with timestamps,
log levels, and originating module details.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

DEFAULT_LOG_FILE = os.path.join(LOG_DIR, f"blt_{datetime.now().strftime('%Y%m%d')}.log")

FORMATTER = logging.Formatter(
    fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(
    name: str = "BLT",
    log_file: Optional[str] = DEFAULT_LOG_FILE,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create or retrieve a structured logger with console and file output.

    Args:
        name: Name of the logger (typically module name).
        log_file: Path to destination log file. If None, file logging is disabled.
        level: Logging level threshold.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger was already created
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(FORMATTER)
        logger.addHandler(console_handler)

        # File handler
        if log_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(FORMATTER)
            logger.addHandler(file_handler)

    return logger


# Default application logger
logger = get_logger("BLT")
