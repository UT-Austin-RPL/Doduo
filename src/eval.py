import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` above is optional line to make environment more convenient
# should be placed at the top of each entry file
#
# main advantages:
# - allows you to keep all entry files in "src/" without installing project as a package
# - launching python file works no matter where is your current work dir
# - automatically loads environment variables from ".env" if exists
#
# how it works:
# - `setup_root()` above recursively searches for either ".git" or "pyproject.toml" in present
#   and parent dirs, to determine the project root dir
# - adds root dir to the PYTHONPATH (if `pythonpath=True`), so this file can be run from
#   any place without installing project as a package
# - sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
#   to make all paths always relative to project root
# - loads environment variables from ".env" in root dir (if `dotenv=True`)
#
# you can remove `pyrootutils.setup_root(...)` if you:
# 1. either install project as a package or move each entry file to the project root dir
# 2. remove PROJECT_ROOT variable from paths in "configs/paths/default.yaml"
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

import logging
from typing import List, Tuple

import hydra
import torch
from omegaconf import DictConfig

from src.utils.pylogger import setup_logger


def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    from src.utils import utils

    log = logging.getLogger("base")

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    log.info("Instantiating loggers...")
    loggers = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating model <{cfg.model._target_}>")
    if cfg.trainer.device is not None:
        device = torch.device(cfg.trainer.device)
    else:
        device = None
    model = hydra.utils.instantiate(
        cfg.model, datamodule=datamodule, loggers=loggers, device=device
    )

    object_dict = {"cfg": cfg, "datamodule": datamodule, "model": model, "logger": loggers}

    if loggers:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)
    model.test(cfg)


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    setup_logger(name="base", level=logging.INFO, log_dir=cfg.paths.output_dir)
    # train the model
    evaluate(cfg)


if __name__ == "__main__":
    main()
