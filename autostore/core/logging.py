# Corporation: HKCLR 
# Project: AutoStore picking robots
# Author: Chang Liu
# Some codes are refered from https://github.com/facebookresearch/pycls


"""Setup loggers for other modules."""


import logging
import sys
from typing import Optional


_FORMAT = "[%(filename)s: %(lineno)3d]: %(message)s"


def setup_logging(log_path: Optional[str] = None) -> None:
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
    }
    if log_path:
        log_handler = logging.FileHandler(log_path)
        stream_handler = logging.StreamHandler(sys.stdout)
        logging_cfg.update({"handlers": [log_handler, stream_handler]})
    else:
        logging_cfg.update({"stream": sys.stdout})
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