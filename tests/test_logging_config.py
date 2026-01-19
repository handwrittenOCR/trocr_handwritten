import pytest
import logging
from pathlib import Path
import tempfile

from trocr_handwritten.utils.logging_config import (
    setup_logging,
    get_logger,
    set_level,
    DEFAULT_LEVEL,
)


class TestLoggingConfig:
    """Tests for the logging configuration module."""

    @pytest.fixture(autouse=True)
    def reset_logging(self):
        """Reset logging state before each test."""
        import trocr_handwritten.utils.logging_config as lc

        lc._configured = False

        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

        yield

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a Logger instance."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_get_logger_configures_once(self):
        """Test that logging is only configured once."""
        import trocr_handwritten.utils.logging_config as lc

        assert lc._configured is False
        get_logger("test1")
        assert lc._configured is True
        get_logger("test2")
        assert lc._configured is True

    def test_setup_logging_creates_console_handler(self):
        """Test that setup_logging creates a console handler."""
        setup_logging()

        root = logging.getLogger()
        handler_types = [type(h).__name__ for h in root.handlers]

        assert "StreamHandler" in handler_types

    def test_setup_logging_with_log_file(self):
        """Test that setup_logging creates a file handler when log_file is provided."""
        import trocr_handwritten.utils.logging_config as lc

        lc._configured = False

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            setup_logging(log_file=log_file)

            root = logging.getLogger()
            handler_types = [type(h).__name__ for h in root.handlers]

            assert "FileHandler" in handler_types
            assert log_file.exists()

    def test_setup_logging_with_log_dir(self):
        """Test that setup_logging creates auto-named log file in directory."""
        import trocr_handwritten.utils.logging_config as lc

        lc._configured = False

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            setup_logging(log_dir=log_dir)

            log_files = list(log_dir.glob("run_*.log"))
            assert len(log_files) == 1

    def test_set_level_changes_level(self):
        """Test that set_level changes the logging level."""
        setup_logging(level=logging.INFO)
        set_level(logging.DEBUG)

        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_setup_logging_respects_level(self):
        """Test that setup_logging respects the provided level."""
        setup_logging(level=logging.WARNING)

        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_default_level_is_info(self):
        """Test that default level is INFO."""
        assert DEFAULT_LEVEL == logging.INFO


class TestLoggerOutput:
    """Tests for actual logging output."""

    @pytest.fixture(autouse=True)
    def reset_logging(self):
        """Reset logging state before each test."""
        import trocr_handwritten.utils.logging_config as lc

        lc._configured = False

        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

        yield

    def test_logger_writes_to_file(self):
        """Test that logger actually writes to file."""
        import trocr_handwritten.utils.logging_config as lc

        lc._configured = False

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_logging(log_file=log_file, name="test_writer")
            logger.info("Test message")

            for handler in logging.getLogger().handlers:
                handler.flush()

            content = log_file.read_text()
            assert "Test message" in content
