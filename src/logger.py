import logging  #to log all errors and exceptions into a text file
import os
from datetime import datetime

# Create a directory for logs if it doesn't exist
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)  # create a directory for logs if it doesn't exist

# Configure logging
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='%(asctime)s %(message)s',                #refer to python logging documentation for more information
    level=logging.INFO
)

# if __name__ == "__main__":   #for testing the logger
#     logging.info("Logging has been configured.")
#     logging.info("This is an info message.")
