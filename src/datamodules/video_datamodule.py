from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import src.utils.dist as dist_utils
from src.utils.hparams import save_hyperparameters


class VideoDataModule:
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_train: Dataset = None,
        data_val: Dataset = None,
        data_test: Dataset = None,
        train_batch_size: int = 64,
        val_batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        save_hyperparameters(self, ignore=["data_train", "data_val", "data_test"])

        self.data_train: Optional[Dataset] = data_train
        self.data_val: Optional[Dataset] = data_val
        self.data_test: Optional[Dataset] = data_test

    def train_dataloader(self):
        if dist_utils.is_distributed():
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.data_train, shuffle=True, rank=dist_utils.get_rank()
            )
        else:
            train_sampler = None

        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=train_sampler is None,
            sampler=train_sampler,
        )

    def val_dataloader(self):
        # repeat validation is ok
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        if dist_utils.get_rank() == 0:
            return DataLoader(
                dataset=self.data_test,
                batch_size=self.hparams.val_batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
            )
        else:
            return None


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "mnist.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
