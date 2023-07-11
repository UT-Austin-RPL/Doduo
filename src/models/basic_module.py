import logging
import os
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from src.models.components.eval import compute_tapvid_metrics, get_track_by_feat
from src.models.components.utils import fig2numpy, tensor2image
from src.utils import dist as dist_utils
from src.utils.utils import data2device

log = logging.getLogger("base")

# correspondence learning module
class BasicModule:
    def __init__(self, net, datamodule, loggers, device):
        super().__init__()
        self.net = net
        self.datamodule = datamodule
        self.loggers = loggers

        if len(loggers) >= 1:
            self.logger = loggers[0]
        else:
            self.logger = None
            log.warning("No logger is set for this module")
        self.cur_step = 0
        self.cur_epoch = 0
        if dist_utils.is_distributed():
            self.device = torch.device(dist_utils.get_rank())
            self.net.to(self.device)
            self.net = DDP(self.net, device_ids=[dist_utils.get_rank()])
            self.distributed = True
        else:
            self.net = self.net.to(device)
            self.device = device
            self.distributed = False

    def on_train_start(self):
        raise NotImplementedError("Please Implement this method")

    def training_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError("Please Implement this method")

    def training_epoch_end(self, outputs: List[Any]):
        raise NotImplementedError("Please Implement this method")

    def validation_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError("Please Implement this method")

    def validation_epoch_end(self, outputs: List[Any]):
        raise NotImplementedError("Please Implement this method")

    def test_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError("Please Implement this method")

    def test_epoch_end(self, outputs: List[Any]):
        raise NotImplementedError("Please Implement this method")

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        self.optimizer = self.hparams.optimizer(
            params=filter(lambda p: p.requires_grad, self.net.parameters())
        )
        if self.hparams.scheduler is not None:
            self.scheduler = self.hparams.scheduler(optimizer=self.optimizer)

    def save_ckpt(self, ckpt_path):
        ckpt = {
            "cur_step": self.cur_step,
            "cur_epoch": self.cur_epoch,
            "state_dict": self.net.module.state_dict()
            if isinstance(self.net, DDP)
            else self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(ckpt, ckpt_path)

    def clean_ckpt(self, keep=["last.ckpt"]):
        all_ckpts = os.listdir(self.ckpt_dir)
        for ckpt in keep:
            if ckpt in all_ckpts:
                all_ckpts.remove(ckpt)
        for ckpt in all_ckpts:
            os.remove(os.path.join(self.ckpt_dir, ckpt))

    def load_ckpt(self, ckpt_path=None, model_only=False, partial=False):
        if ckpt_path is None:
            return
        else:
            ckpt = torch.load(ckpt_path)
            if isinstance(self.net, DDP):
                out = self.net.module.load_state_dict(ckpt["state_dict"], strict=not partial)
            else:
                out = self.net.load_state_dict(ckpt["state_dict"], strict=not partial)
            log.info(f"{out}")
            if not model_only:
                if "optimizer" in ckpt and hasattr(self, "optimizer"):
                    self.optimizer.load_state_dict(ckpt["optimizer"])
                else:
                    log.info(
                        f"Optimizer not loaded, ckpt: {'optimizer' in ckpt}, self: {hasattr(self, 'optimizer')}"
                    )
                if "scheduler" in ckpt and hasattr(self, "scheduler"):
                    self.scheduler.load_state_dict(ckpt["scheduler"])
                else:
                    log.info(
                        f"Scheduler not loaded, ckpt: {'scheduler' in ckpt}, self: {hasattr(self, 'scheduler')}"
                    )
                self.cur_step = ckpt.get("cur_step", 0)
                self.cur_epoch = ckpt.get("cur_epoch", 0)
            del ckpt
            torch.cuda.empty_cache()

    def train(self, cfg):
        log.info("Starting training...")
        self.cfg = cfg
        self.on_train_start()
        self.net.train()
        self.configure_optimizers()
        if cfg.get("ckpt_path"):
            log.info(f"Resume training from {cfg.ckpt_path}")
            self.load_ckpt(cfg.ckpt_path)
        elif cfg.get("pre_ckpt_path"):
            log.info(f"Load pretrained model from {cfg.pre_ckpt_path}")
            self.load_ckpt(cfg.pre_ckpt_path, model_only=True)
        elif cfg.get("partial_ckpt_path"):
            log.info(f"Load partial model from {cfg.partial_ckpt_path}")
            self.load_ckpt(cfg.partial_ckpt_path, model_only=True, partial=True)

        train_dl = self.datamodule.train_dataloader()
        for epoch in range(self.cur_epoch, cfg.trainer.max_epochs):
            if self.distributed:
                train_dl.sampler.set_epoch(epoch)
            train_outputs = []
            for batch_idx, batch in enumerate(train_dl):
                batch = data2device(batch, self.device)
                train_outputs.append(self.training_step(batch, batch_idx))
                self.cur_step += 1
                if (
                    self.cfg.trainer.get("fast_dev_run")
                    or self.cur_step > self.cfg.trainer.max_epochs * self.train_size
                ):
                    break
            self.training_epoch_end(train_outputs)

            if (
                epoch % self.cfg.trainer.val_freq == 0
                or self.cur_step > self.cfg.trainer.max_epochs * self.train_size
            ):
                self.net.eval()
                val_outputs = []
                for batch_idx, batch in enumerate(self.datamodule.val_dataloader()):
                    batch = data2device(batch, self.device)
                    val_outputs.append(self.validation_step(batch, batch_idx))
                    if self.cfg.trainer.get("fast_dev_run"):
                        break
                self.validation_epoch_end(val_outputs)
                self.net.train()

            self.cur_epoch += 1
            if self.cfg.trainer.get("fast_dev_run"):
                break

        log.info("Training finished.")

    def test(self, cfg):
        log.info("Starting testing...")
        if cfg.get("train"):
            log.info("Test after training")
        else:
            log.info("Test only")
            assert os.path.isfile(cfg.ckpt_path), "Must specify a valid ckpt_path for test only"
            self.load_ckpt(cfg.ckpt_path)

        self.cfg = cfg
        test_outputs = []
        self.net.eval()
        for batch_idx, batch in enumerate(self.datamodule.test_dataloader()):
            batch = data2device(batch, self.device)
            test_outputs.append(self.test_step(batch, batch_idx))
            if self.cfg.trainer.get("fast_dev_run"):
                break
        self.test_epoch_end(test_outputs)
        log.info("Test finished.")

    def log(self, metric_name, metric_value):
        for logger in self.loggers:
            logger.log_metrics({metric_name: metric_value}, step=self.cur_step)


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
