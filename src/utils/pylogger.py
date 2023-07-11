import logging
import os

from src.utils.dist import get_rank


def setup_logger(name="base", level=logging.INFO, log_dir=None, screen=True) -> logging.Logger:
    """set up logger."""
    lg = logging.getLogger(name)
    formatter = logging.Formatter(
        f"rank: {get_rank()} "
        + "%(asctime)s.%(msecs)03d - %(pathname)s:%(lineno)d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    lg.setLevel(level)
    if not (log_dir is None):
        log_file = f"{log_dir}/rank_{get_rank()}.log"
        print(f"Logging to {os.path.abspath(log_file)}")
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen and get_rank() == 0:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)
    return lg
