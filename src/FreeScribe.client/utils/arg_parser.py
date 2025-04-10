import argparse
from utils.log_config import logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-debug", help="Enable the file logger")
    args = parser.parse_args()

    logger.info(args)
