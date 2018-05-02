import logging
from sys import stdout

def setup_custom_logger(name):
    formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S')

    handler = logging.StreamHandler(stdout)
    # handler = logging.FileHandler('/home/shadi/PycharmProjects/pyndl/pyndl/logs/hello.log')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger

setup_custom_logger('pyndl')