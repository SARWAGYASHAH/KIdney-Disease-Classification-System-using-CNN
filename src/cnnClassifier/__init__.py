import os
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")# file name 
os.makedirs(log_dir, exist_ok=True)## directary is made

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,# this is the format that is being logged 

    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("cnnClassifierLogger")## logger object is created and the name is written down 