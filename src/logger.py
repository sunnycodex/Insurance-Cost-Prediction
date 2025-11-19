import logging
import os
from datetime import datetime

# Create timestamp for log filename
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# Define logs directory path
logs_path = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_path, exist_ok=True)

# Complete log file path
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure logging with format and output file
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format='[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s'
)