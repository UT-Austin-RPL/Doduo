import logging
import os
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning
import torch
from einops import rearrange
from PIL import Image
from torchmetrics import MaxMetric, MeanMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy

import src.utils.dist as dist_utils
from src.models.basic_module import BasicModule
from src.models.components.eval import compute_tapvid_metrics
from src.models.components.unimatch.geometry import coords_grid, flow_warp
from src.models.components.utils import vis_train, vis_val
from src.utils.hparams import save_hyperparameters

log = logging.getLogger("base")

# correspondence learning module
class CorrFlowModuleV0(BasicModule):
    def __init__(
        self,
        net: torch.nn.Module,
        datamodule: Any,
        loggers: pytorch_lightning.loggers.base.LightningLoggerBase,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        eval_size: int = 256,
        num_vis: int = 4,
        log_freq: int = 100,
        mixed_precision: bool = False,
        bidirectional: bool = False,
        cycle_consistency: bool = False,
        eval_visible_thresh: float = -1,  # no eval visible thresh
        **kwargs,
    ):
        super().__init__(net, datamodule, loggers, device)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        save_hyperparameters(self, ignore=["net", "datamodule"])

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.val_jaccard = MeanMetric().to(self.device)
        self.val_avg_dist = MeanMetric().to(self.device)

        # for averaging loss across batches
        self.train_loss = MeanMetric().to(self.device)

        # for tracking best so far validation accuracy
        self.val_jaccard_best = MaxMetric().to(self.device)
        self.best_epoch = 0

        # mixed precision
        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

    def forward(self, x: torch.Tensor):
        return self.net.forward_features(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_jaccard_best doesn't store accuracy from these checks
        self.val_jaccard_best.reset()
        self.train_vis_dir = f"{self.cfg.paths.output_dir}/train_vis"
        self.val_vis_dir = f"{self.cfg.paths.output_dir}/val_vis"
        os.makedirs(self.train_vis_dir, exist_ok=True)
        self.ckpt_dir = f"{self.cfg.paths.output_dir}/checkpoints"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        # select idxs to visualize
        self.train_size = len(self.datamodule.train_dataloader())
        self.train_vis_idxs = np.linspace(
            0, self.train_size - 1, num=self.hparams.num_vis, dtype=int
        )
        log.info(f"Save output to {os.path.abspath(self.cfg.paths.output_dir)}")
        self.wandb_logger = None
        for logger in self.loggers:
            if isinstance(logger, pytorch_lightning.loggers.wandb.WandbLogger):
                self.wandb_logger = logger
                self.wandb_logger.log_text(
                    key="train_log", columns=["train_log"], data=[[self.cfg.paths.output_dir]]
                )

    def train_step(self, batch: Any):
        # train step
        # reconstruction
        return self.net(
            batch,
            bidirectional=self.hparams.bidirectional,
            cycle_consistency=self.hparams.cycle_consistency,
        )

    def val_step(self, batch: Any):
        # val step
        # feature extraction
        frames = batch["frames"]
        query_points = batch["query_points"]  # B * N * 3, (t, y, x)
        B, N = query_points.shape[:2]
        _, T, _, H, W = frames.shape
        eval_visible_thresh = self.hparams.eval_visible_thresh * np.sqrt(H * W)
        # B * N
        start_frame_idxs = query_points[:, :, 0].long()
        start_frames = frames[torch.arange(B), start_frame_idxs]  # B * N * C * H * W
        start_frames = rearrange(start_frames, "b n c h w -> (b n) c h w")

        query_points_norm = query_points.float()[:, :, 1:] * 2 - 1
        # (B * N) * 1 * 1 * 2
        query_points_norm = rearrange(query_points_norm, "b n c -> (b n) 1 1 c")

        match_results = torch.zeros(B, N, T, 2).to(self.device)
        pred_occ = torch.zeros(B, N, T).to(self.device)
        sim_maps = []
        for t in range(T):
            target_frames = frames[:, [t]].repeat(1, N, 1, 1, 1)  # B * N * C * H * W
            target_frames = rearrange(target_frames, "b n c h w -> (b n) c h w")
            orig_batch_size = target_frames.shape[0]
            # (B * N) * C * H * W
            _, vis_dict = self.net(
                {"frame_src": start_frames, "frame_dst": target_frames},
                return_feature=True,
                cycle_consistency=True,
            )
            # (B * N) * 2 * H * W
            flow = vis_dict["flow"][:orig_batch_size]
            backward_flow = vis_dict["flow"][orig_batch_size:]
            # (B * N) * 1 * H * W
            cycle_dist = (flow_warp(backward_flow, flow) + flow).norm(dim=1, p=2, keepdim=True)

            coords = coords_grid(flow.size(0), flow.size(2), flow.size(3), device=flow.device)
            target_points = coords + flow
            query_results = torch.nn.functional.grid_sample(
                target_points, query_points_norm.flip(-1), align_corners=False
            )
            query_cycle_dist = torch.nn.functional.grid_sample(
                cycle_dist, query_points_norm.flip(-1), align_corners=False
            )
            # B * N * 2
            query_results = (
                rearrange(query_results, "(b n) c h w -> b n c h w", b=B).squeeze(-1).squeeze(-1)
            )
            # B * N * 1
            query_cycle_dist = (
                rearrange(query_cycle_dist, "(b n) c h w -> b n c h w", b=B)
                .squeeze(-1)
                .squeeze(-1)
            )
            # normalize to [0, 1]
            query_results = query_results / torch.tensor([W, H]).to(self.device)
            match_results[:, :, t] = query_results
            pred_occ[:, :, t] = query_cycle_dist.squeeze(-1) > eval_visible_thresh

            # get similarity map
            start_frame_feats, target_frames_feats = (
                vis_dict["feature0"][:orig_batch_size],
                vis_dict["feature1"][:orig_batch_size],
            )
            query_feats = torch.nn.functional.grid_sample(
                start_frame_feats, query_points_norm.flip(-1), align_corners=False
            )
            # B * N * c
            query_feats = (
                rearrange(query_feats, "(b n) c h w -> b n c h w", b=B).squeeze(-1).squeeze(-1)
            )
            target_frames_feats = rearrange(target_frames_feats, "(b n) c h w -> b n c h w", b=B)
            sim_map = torch.einsum("bnc,bnchw->bnhw", query_feats, target_frames_feats)
            sim_maps.append(sim_map)
        similarity = torch.stack(sim_maps, dim=2)  # B * N * T * H * W
        return match_results, similarity, pred_occ

    def extract_features(self, frame_src: torch.Tensor, frame_dst: torch.Tensor):
        if self.distributed:
            return self.net.module.forward_features(frame_src, frame_dst)
        else:
            return self.net.forward_features(frame_src, frame_dst)

    def training_step(self, batch: Any, batch_idx: int):
        self.optimizer.zero_grad()
        if self.hparams.mixed_precision:
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                loss_dict, pred_dict = self.train_step(batch)
                loss = 0.0
                for k, v in loss_dict.items():
                    weight = getattr(self.hparams, f"loss_{k}_weight")
                    if weight > 0:
                        loss += v * weight
                    self.log(f"train/loss_{k}", v)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            loss_dict, pred_dict = self.train_step(batch)
            loss = 0.0
            for k, v in loss_dict.items():
                weight = getattr(self.hparams, f"loss_{k}_weight")
                if weight > 0:
                    loss += v * weight
                self.log(f"train/loss_{k}", v)
            loss.backward()
            self.optimizer.step()

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", loss)
        if self.cur_step % self.hparams.log_freq == 0:
            log.info(
                f"Epoch [{self.cur_epoch}/{self.cfg.trainer.max_epochs}] Step {self.cur_step} {self.cur_step % self.train_size}/{self.train_size}: loss = {self.train_loss.compute()}, lr = {self.optimizer.param_groups[0]['lr']:.2e}"
            )

        # log the predicted images with wandb
        if dist_utils.get_rank() == 0 and batch_idx in self.train_vis_idxs:
            fig_img = vis_train(batch, pred_dict, to_lab=self.hparams.get("lab", False))
            fig_img.save(
                f"{self.train_vis_dir}/{self.cur_epoch}_{batch_idx:010d}_{batch['frame_src_name'][0]}_{batch['frame_dst_name'][0]}.png"
            )
            if self.wandb_logger is not None:
                self.wandb_logger.log_image(key=f"train_vis/{batch_idx:010d}", images=[fig_img])
        # dist_utils.dist_barrier()

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        # log learning rate
        train_loss = self.train_loss.compute()
        if dist_utils.get_rank() == 0:
            self.log("train/loss_epoch", train_loss)
            self.log("train/lr", self.optimizer.param_groups[0]["lr"])
        dist_utils.dist_barrier()
        self.train_loss.reset()
        self.scheduler.step()

    @torch.no_grad()
    def validation_step(self, batch: Any, batch_idx: int):
        # B x T x C x H x W
        tracks, similarity, pred_occ = self.val_step(batch)

        query_points = batch["query_points"].cpu().numpy()
        query_points[:, :, 1:] *= self.hparams.eval_size
        gt_occluded = batch["occluded"].cpu().bool().numpy()
        gt_target_points = batch["target_points"].cpu().numpy() * self.hparams.eval_size
        if self.hparams.eval_visible_thresh > 0:
            pred_occ_np = pred_occ.cpu().numpy()
        else:
            # if eval_visible_thresh <= 0, use the ground truth occluded
            pred_occ_np = gt_occluded

        tracks_np = tracks.cpu().numpy() * self.hparams.eval_size
        eval_results = compute_tapvid_metrics(
            query_points, gt_occluded, gt_target_points, pred_occ_np, tracks_np, "first"
        )
        # update and log metrics
        self.val_jaccard(eval_results["average_jaccard"])
        self.val_avg_dist(eval_results["avg_distance"])
        # log images
        if dist_utils.get_rank() == 0:
            fig_imgs = vis_val(batch, similarity, tracks, to_lab=self.hparams.get("lab", False))
            video_name = batch["video_name"][0]
            for n_point, fig_img in enumerate(fig_imgs):
                save_dir = f"{self.val_vis_dir}/{video_name}/{n_point:03d}/"
                os.makedirs(save_dir, exist_ok=True)
                fig_img.save(f"{save_dir}/{self.cur_epoch:04d}_{self.cur_step:08d}.png")
            if self.wandb_logger is not None and video_name in self.hparams.val_vis_list_wandb:
                self.wandb_logger.log_image(key=f"val_vis/{video_name}", images=[fig_imgs[0]])
        return None

    def validation_epoch_end(self, outputs: List[Any]):
        jaccard = self.val_jaccard.compute()
        avg_dist = self.val_avg_dist.compute()
        prev_jaccard = self.val_jaccard_best.compute()
        self.val_jaccard_best(jaccard)
        jaccard_best = self.val_jaccard_best.compute()
        if dist_utils.get_rank() == 0:
            if not self.cfg.trainer.get("fast_dev_run"):
                last_save_ckpt_path = f"{self.ckpt_dir}/last.ckpt"
                log.info(f"Saving checkpoint to {last_save_ckpt_path}")
                self.save_ckpt(last_save_ckpt_path)
            # log `val_jaccard_best` as a value through `.compute()` method, instead of as a metric object
            # otherwise metric would be reset by lightning after each epoch
            self.log("val/jaccard_best", jaccard_best)
            self.log("val/jaccard", jaccard)
            self.log("val/avg_distance", avg_dist)
            log.info(
                f"Epoch [{self.cur_epoch}/{self.cfg.trainer.max_epochs}]: jaccard = {jaccard:.4f}, avg_distance = {avg_dist:.4f}, jaccard_best = {jaccard_best:.4f} at epoch {self.best_epoch}"
            )
        dist_utils.dist_barrier()
        self.val_jaccard.reset()
        self.val_avg_dist.reset()

    @torch.no_grad()
    def test_step(self, batch: Any, batch_idx: int):
        assert dist_utils.get_rank() == 0, "test should only be run on rank 0"
        tracks, similarity, pred_occ = self.val_step(batch)
        query_points = batch["query_points"].cpu().numpy()
        query_points[:, :, 1:] *= self.hparams.eval_size
        gt_occluded = batch["occluded"].cpu().bool().numpy()
        gt_target_points = batch["target_points"].cpu().numpy() * self.hparams.eval_size
        if self.hparams.eval_visible_thresh > 0:
            pred_occ_np = pred_occ.cpu().numpy()
        else:
            # if eval_visible_thresh <= 0, use the ground truth occluded
            pred_occ_np = gt_occluded
        tracks = tracks.cpu().numpy() * self.hparams.eval_size
        eval_results = compute_tapvid_metrics(
            query_points, gt_occluded, gt_target_points, pred_occ_np, tracks, "first"
        )
        return eval_results

    def test_epoch_end(self, outputs: List[Any]):
        # get average results
        avg_results = {k: [] for k in outputs[0].keys()}
        for eval_results in outputs:
            for k, v in eval_results.items():
                avg_results[k].append(v)
        for k, v in avg_results.items():
            avg_results[k] = np.mean(v)
        jaccard = avg_results["average_jaccard"]
        avg_dist = avg_results["avg_distance"]
        average_pts_within_thresh = avg_results["average_pts_within_thresh"]
        occlusion_accuracy = avg_results["occlusion_accuracy"]

        self.log("test/jaccard", jaccard)
        self.log("test/avg_distance", avg_dist)
        self.log("test/average_pts_within_thresh", average_pts_within_thresh)
        self.log("test/occlusion_accuracy", occlusion_accuracy)
        log.info(f"test jaccard: {jaccard}")
        log.info(f"test avg distance: {avg_dist}")
        log.info(f"test average_pts_within_thresh: {average_pts_within_thresh}")
        log.info(f"test occlusion_accuracy: {occlusion_accuracy}")


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
