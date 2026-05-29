
import logging
import sys
from pathlib import Path
from datetime import datetime

def get_logger(name: str, log_dir: str = "./log_dirs/logs.log") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers if already set
        return logger

    logger.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(Path(log_dir) / f"{name}_{timestamp}.log")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger