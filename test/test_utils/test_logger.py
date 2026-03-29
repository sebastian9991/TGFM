import logging

import pytest

from tgfm.utils.logger import setup_logging


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset the logging configuration before and after each test."""
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    root.setLevel(logging.WARNING)
    yield
    for handler in root.handlers[:]:
        root.removeHandler(handler)


def test_setup_logging_levels():
    """Verify and print root handlers and their levels."""
    setup_logging(stream_logging_level=logging.INFO)
    root = logging.getLogger()

    for i, h in enumerate(root.handlers):
        print(f'Handler {i}: {h}, Level: {h.level}')

    assert root.handlers[0].level == logging.INFO


def test_setup_logging_stream_only():
    """Verify that only the StreamHandler is added by default."""
    setup_logging()

    root_logger = logging.getLogger()
    handlers = root_logger.handlers

    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.StreamHandler)
    assert handlers[0].level == logging.INFO


def test_setup_logging_with_file(tmp_path):
    """Verify that both StreamHandler and FileHandler are added when path is provided."""
    log_file = tmp_path / "test.log"
    setup_logging(log_file_path=str(log_file), log_file_logging_level=logging.ERROR)

    root_logger = logging.getLogger()
    handlers = root_logger.handlers

    logging.info('Testing setup logger with file.')


    assert len(handlers) == 2

    file_handlers = [h for h in handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
    assert file_handlers[0].baseFilename == str(log_file)
    assert file_handlers[0].level == logging.ERROR

def test_logging_output_to_file(tmp_path):
    """Verify that logs are actually written to the file."""
    log_file = tmp_path / "output.log"
    setup_logging(log_file_path=str(log_file), log_file_logging_level=logging.DEBUG)

    test_message = "This is a test debug message"
    logging.debug(test_message)

    for handler in logging.getLogger().handlers:
        handler.close()

    assert log_file.exists()
    with open(log_file, "r") as f:
        content = f.read()
        print(content)
        assert test_message in content
        assert "DEBUG" in content

def test_matplotlib_suppression():
    """Verify that matplotlib loggers are set to WARNING."""
    setup_logging()

    assert logging.getLogger("matplotlib").level == logging.WARNING
    assert logging.getLogger("matplotlib.font_manager").level == logging.WARNING

def test_custom_levels():
    """Verify that custom logging levels are respected."""
    setup_logging(stream_logging_level=logging.CRITICAL)

    root_logger = logging.getLogger()
    stream_handler = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)][0]

    assert stream_handler.level == logging.CRITICAL
