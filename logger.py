# logger.py
import logging
import os
from config import LOG_FILE_PATH, LOG_TO_CONSOLE

def setup_logger():
    """
    Sets up a logger to output messages to a file and optionally to the console.
    Ensures handlers are not duplicated on successive calls (e.g., Streamlit reruns).
    """
    logger = logging.getLogger("socratic_tutor")
    logger.setLevel(logging.INFO) # Set the minimum level for logging

    # Prevent adding duplicate handlers on Streamlit reruns
    if not logger.handlers: # Check if handlers already exist
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # File handler
        file_handler = logging.FileHandler(LOG_FILE_PATH)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler (optional)
        if LOG_TO_CONSOLE:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

    return logger

# Initialize the logger when this module is imported
logger = setup_logger()
