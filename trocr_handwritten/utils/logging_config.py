import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FORMAT_SIMPLE = "%(levelname)s - %(message)s"
DEFAULT_LEVEL = logging.INFO

_configured = False


def setup_logging(
    level: int = DEFAULT_LEVEL,
    log_file: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    name: Optional[str] = None,
) -> logging.Logger:
    """
    Configure logging for the application.

    Args:
        level: Logging level (default: INFO).
        log_file: Optional specific log file path.
        log_dir: Optional directory for auto-named log files.
        name: Optional logger name for the returned logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    global _configured

    if not _configured:
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root_logger.addHandler(console_handler)

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
            root_logger.addHandler(file_handler)
        elif log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"run_{timestamp}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
            root_logger.addHandler(file_handler)

        _configured = True

    return logging.getLogger(name)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    Initializes default configuration if not already done.

    Args:
        name: Logger name (typically __name__).

    Returns:
        logging.Logger: Logger instance.
    """
    global _configured

    if not _configured:
        setup_logging()

    return logging.getLogger(name)


def set_level(level: int) -> None:
    """
    Change the logging level for all handlers.

    Args:
        level: New logging level.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)
