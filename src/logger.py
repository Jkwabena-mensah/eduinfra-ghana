import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(name: str, log_dir: str = 'logs') -> logging.Logger:
    """Creates a professional logger that always points to the project root logs."""
    # This finds the directory above 'src', which is the project root (C:\Dev\eduinfra-ghana)
    project_root = Path(__file__).parent.parent.absolute()
    full_log_dir = project_root / log_dir
    full_log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Console Handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S'))

        # File Handler (UTF-8)
        log_file = full_log_dir / f"eduinfra_{datetime.now():%Y%m%d}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))

        logger.addHandler(console)
        logger.addHandler(file_handler)

    return logger