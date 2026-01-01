"""
Simple logging helper for the IVF project.
"""

import logging
from pathlib import Path
from typing import Optional


def configure_logging(log_path: Path, level: int = logging.INFO) -> logging.Logger:
    """
    Configure a unified logger with file and console handlers.
    """
    logger = logging.getLogger("ivf")
    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path) for h in logger.handlers):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not any(type(h) is logging.StreamHandler for h in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Retrieve a configured logger.
    """
    logger_name = name or "ivf"
    return logging.getLogger(logger_name)
