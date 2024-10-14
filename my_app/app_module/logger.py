import logging

# Basic configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Now you can log events
logging.info("This is an info message")
logging.warning("This is a warning message")
logging.error("This is an error message")