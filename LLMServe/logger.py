import logging
import os
from logging.handlers import RotatingFileHandler
import sys

_logger_instance = None
_console_handler = None


def get_console_handler():
    global _console_handler
    if _console_handler is None:
        _console_handler = logging.StreamHandler(sys.stdout)
    return _console_handler


def setup_local_logger(log_file_path, caller_file, console_level=logging.INFO, file_level=logging.DEBUG):
    logger = logging.getLogger(caller_file + "_logger")
    logger.setLevel(logging.DEBUG)

    console_handler = get_console_handler()
    # console_handler.setLevel(console_level)
    assert console_handler.level == console_level

    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setLevel(file_level)

    formatter = logging.Formatter(
        "[%(levelname)s] [%(asctime)s]: %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Local logger from [{caller_file}] setup in [{log_file_path}]")

    return logger


def setup_logger(log_file_path, console_level=logging.INFO, file_level=logging.DEBUG):

    global _logger_instance
    global _console_handler

    if _logger_instance is not None:
        return _logger_instance

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if not logger.hasHandlers():
        console_handler = get_console_handler()
        console_handler.setLevel(console_level)

        file_handler = RotatingFileHandler(
            log_file_path, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setLevel(file_level)

        formatter = logging.Formatter(
            "[%(levelname)s] [%(asctime)s]  [%(filename)s] : %(message)s"
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        _logger_instance = logger

        logger.info(f"Global logger setup in [{log_file_path}]")

    return logger


def init_logger(
    log_file_path=None,
    # log_file_path=os.path.join(
    #     os.path.join(os.path.dirname(__file__), "../results/"), "run.log"
    # ),
    console_level=logging.WARNING,
    file_level=logging.DEBUG,
):
    logger = setup_logger(log_file_path)
    return logger


if __name__ == "__main__":
    logger = init_logger()
    logger.info("This is an info log message.")
    logger.error("This is an error log message.")
