# Corporation: HKCLR 
# Project: AutoStore picking robots
# Author: Chang Liu
# Date last modified: May 22, 2023
# Some codes are referenced from https://github.com/facebookresearch/pycls


"""Setup loggers for other modules."""


import logging
import sys


_FORMAT = "[%(filename)s: %(lineno)3d]: %(message)s"


def setup_logging() -> None:
    """Set up root logger by add a StreamHandler with a
    Formatter on it.

    Args:
        name (str): 
            Name of the logger, which is recommended to set as
            the ""__name__"" attribute of the calling module.
    
    Returns:
        logger (logging.Logger):
            A Logger instance.
    """
    # Clear the root logger to prevent any existing logging config
    # (e.g. set by another module) from messing with our setup
    logging.root.handlers = []
    logging_cfg = {
        "level": logging.INFO, 
        "format": _FORMAT,
        "stream": sys.stdout
    }
    logging.basicConfig(**logging_cfg)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with a specified name.
    
    Args:
        name (str):
            The name of logger, which is recommended to use 
            "__name__" of the importing module.
    
    Returns:
        logger (logging.Logger):
            A python built-in Logger instance with
            the specified name.
    """
    return logging.getLogger(name)