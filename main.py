import logging
import os

from trainer import Trainer

LOG_DIRECTORY= 'logs'
LOG_FILE = os.path.join(LOG_DIRECTORY, 'history.log')
MAX_EPOCHS = 100

logger = logging.getLogger()

def configure_logging():
    os.makedirs(LOG_DIRECTORY, exist_ok=True)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    file_handler.setFormatter(file_format)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def main():
    configure_logging()
    trainer = Trainer()
    trainer.train(MAX_EPOCHS)

if __name__ == '__main__':
    main()
