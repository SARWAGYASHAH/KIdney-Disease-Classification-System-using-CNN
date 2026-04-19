import logging
from pathlib import Path


LOG_DIR = Path("logs")
LOG_FILE_PATH = LOG_DIR / "running_logs.log"
LOG_FORMAT = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("kidney_disease_classifier")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
