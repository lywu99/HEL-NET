
import os
import torch
import logging
import datetime
import math

def get_timestamp():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


def get_logger(name, root, level=logging.INFO, screen=False, tofile=True):
    """get logger"""
    logger = logging.getLogger(name)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    logger.setLevel(level)
    if tofile:
        log_file = os.path.join(root, name + "_{}.log".format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger
