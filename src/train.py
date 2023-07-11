import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import logging
import os
from typing import List, Optional, Tuple

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

import src.utils.dist as dist_utils
from src.utils.pylogger import setup_logger


def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    from src.utils import utils

    log = logging.getLogger("base")

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    if dist_utils.get_rank() == 0:
        log.info("Instantiating loggers...")
        loggers = utils.instantiate_loggers(cfg.get("logger"))
    else:
        loggers = []

    # upload code to wandb
    for logger in loggers:
        if isinstance(logger, pl.loggers.wandb.WandbLogger):
            import wandb

            code_artifact = wandb.Artifact(name="src_code", type="code")
            code_artifact.add_dir(f"{os.getcwd()}/src/")
            wandb.log_artifact(code_artifact)

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

    if cfg.get("train"):
        model.train(cfg)

    if cfg.get("test") and dist_utils.get_rank() == 0:
        model.test(cfg)

    for logger in loggers:
        logger.finalize("success")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    setup_logger(name="base", level=logging.INFO, log_dir=cfg.paths.output_dir)
    # train the model
    train(cfg)


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        dist_utils.dist_init(int(os.environ["LOCAL_RANK"]))
    main()
    dist_utils.dist_destroy()
