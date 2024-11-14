from typing import Optional
import logging
import os

def get_logger(name: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name or __name__)
    log_level = os.getenv("TORCHSTATE_LOG_LEVEL", "INFO")

    logger.setLevel(level=getattr(logging, log_level, logging.INFO))

    handler = logging.StreamHandler()
    logger.addHandler(handler)
    # Adding a formatter to include time, name, and log level
    formatter = logging.Formatter('%(asctime)s %(name)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.propagate = False  # Prevent the log messages from being propagated to the root logger

    return logger
