# config.py

import logging

# Model configuration
MODEL_NAME = "phi3:mini"
TEMPERATURE = 0
SEED = 42
MAX_ATTEMPTS = 3


# Logger setup
def setup_logger() -> logging.Logger:
    """Configure logger with console and file handlers."""
    logger = logging.getLogger("langchain_app")
    logger.setLevel(logging.DEBUG)  # Capture everything at logger level

    # Prevent duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(module)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler - INFO and above only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler - DEBUG and above (captures everything)
    file_handler = logging.FileHandler("run.log", mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()