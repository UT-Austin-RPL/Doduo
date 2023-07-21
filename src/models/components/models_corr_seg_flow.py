import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.models.components.dino import vision_transformer as vits
from src.models.components.dino.utils import load_pretrained_weights
from src.models.components.unimatch.geometry import flow_warp
from src.models.components.unimatch.unimatch import UniMatch, coords_grid


class CorrSegFlowNet(nn.Module):
    def __init__(
        self,
        unimatch: nn.Module,
        dino_backbone: Tuple[str, int],
        dino_corr_mask_ratio: float = 0.1,
        dino_corr_mask_binary: bool = True,
        num_loss_mask: int = 2,
        loss_mask_query_method: str = "correlation",
        **kwargs,
    ):
        super().__init__()
        self.dino_corr_mask_ratio = dino_corr_mask_ratio
        self.dino_corr_mask_binary = dino_corr_mask_binary
        self.unimatch = unimatch
        arch, patch_size = dino_backbone
        self.dino = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
        for k in self.dino.parameters():
            k.requires_grad = False
        load_pretrained_weights(self.dino, "", "teacher", arch, patch_size)
        if not self.dino_corr_mask_binary:
            self.dino_mask_predictor = nn.Linear(1, 1, bias=True)
        self.num_loss_mask = num_loss_mask
        self.loss_mask_query_method = loss_mask_query_method

    def forward(
        self,
        batch: Dict,
        return_feature: bool = False,
        bidirectional: bool = False,
        cycle_consistency: bool = False,
    ):
        assert not (bidirectional and cycle_consistency)
        frame_src_in = batch["frame_src"]
        frame_dst_in = batch["frame_dst"]
        if "frame_src_clean" in batch:
            frame_src = batch["frame_src_clean"]
            frame_dst = batch["frame_dst_clean"]
        else:
            frame_src = frame_src_in
            frame_dst = frame_dst_in

        # clean frame as target
        if bidirectional or cycle_consistency:
            frame_to_warp = torch.cat((frame_dst, frame_src), dim=0)
            frame_target = torch.cat((frame_src, frame_dst), dim=0)
        else:
            frame_to_warp = frame_dst
            frame_target = frame_src

        corr_mask, dino_feat_src, dino_feat_dst = get_dino_corr_mask(
            self.dino,
            frame_target,
            frame_to_warp,
            mask_ratio=self.dino_corr_mask_ratio,
            binary=self.dino_corr_mask_binary,
            return_feat=True,
        )

        if not self.dino_corr_mask_binary:
            # B * (h*w) * (h*w) * 1
            corr_mask = self.dino_mask_predictor(corr_mask.unsqueeze(-1))
            # B * (h*w) * (h*w)
            corr_mask = torch.sigmoid(corr_mask).squeeze(-1)

        flow, flow_low, correlation, feature0, feature1 = self.unimatch(
            frame_src_in,
            frame_dst_in,
            return_feature=True,
            bidirectional=bidirectional,
            cycle_consistency=cycle_consistency,
            corr_mask=corr_mask,
        )
        if torch.isnan(flow).any():
            __import__("pdb").set_trace()

        # no need to concat feat because already done in get_dino_corr_mask
        feature_to_warp = dino_feat_dst
        feature_target = dino_feat_src
        # warp frame and feature
        frame_warped = flow_warp(frame_to_warp, flow)
        feature_warped = flow_warp(feature_to_warp, flow_low)

        if "seg_src" in batch.keys():
            assert not bidirectional and not cycle_consistency
            seg_src = batch["seg_src"]  # B * N * H * W
            # get loss mask
            if self.loss_mask_query_method == "correlation":
                with torch.no_grad():
                    B, _, h, w = feature0.shape
                    H, W = frame_src.shape[2], frame_src.shape[3]
                    correlation_max = correlation.max(-1)[0].view(B, 1, h, w)
                    correlation_max = F.interpolate(correlation_max, size=(H, W))
                    correlation_max_seg = (correlation_max * seg_src).sum(-1).sum(-1) / (
                        seg_src.sum(-1).sum(-1) + 1e-8
                    )
                    # B, self.num_loss_mask, H, W
                    max_seg_corr = seg_src[
                        torch.arange(correlation_max_seg.size(0))[:, None],
                        correlation_max_seg.topk(self.num_loss_mask)[1],
                    ]
                    segs = max_seg_corr
            else:
                raise NotImplementedError(
                    f"loss_mask_query_method {self.loss_mask_query_method} not implemented"
                )

            # B * H * W
            selected_masked_aggregated = segs.sum(1, keepdim=False)
            selected_masked_aggregated = (selected_masked_aggregated > 0).float()
            selected_masked_aggregated_feat = F.interpolate(
                selected_masked_aggregated.unsqueeze(1),
                size=dino_feat_src.shape[2:],
                mode="nearest",
            ).squeeze(1)

            photometric_loss = (
                charbonnier_loss(frame_warped - frame_target).mean(1) * selected_masked_aggregated
            )
            photometric_loss = photometric_loss.sum(dim=(1, 2)) / (
                torch.sum(selected_masked_aggregated, dim=(1, 2)) + 1e-8
            )

            featuremetric_loss = (
                charbonnier_loss(feature_warped - feature_target).mean(1)
                * selected_masked_aggregated_feat
            )
            featuremetric_loss = featuremetric_loss.sum(dim=(1, 2)) / (
                torch.sum(selected_masked_aggregated_feat, dim=(1, 2)) + 1e-8
            )

            smoothness_loss = compute_smoothness_loss(flow, segs)

            distance_consistency_loss = compute_distance_consistency_loss(flow, segs)
            if torch.isnan(smoothness_loss).any() or torch.isnan(distance_consistency_loss).any():
                __import__("pdb").set_trace()

            loss_dict = {
                "smoothness": smoothness_loss.mean(),
                "photometric": photometric_loss.mean(),
                "featuremetric": featuremetric_loss.mean(),
                "distance_consistency": distance_consistency_loss.mean(),
            }

            segs_vis = torch.cat((torch.zeros_like(segs[:, [0], :, :]), segs), dim=1)
            segs_vis = torch.argmax(segs_vis, dim=1) / segs_vis.size(1)
            vis_dict = {
                "selected_mask": selected_masked_aggregated,
                "cycle_similarity": segs_vis,
                "reconstruction": frame_warped,
                "flow": flow,
                "flow_low": flow_low,
                "corr_mask": corr_mask,
                "correlation": correlation,
            }
        else:
            # no segmentation, just infer flow
            loss_dict = None
            vis_dict = {
                "selected_mask": torch.zeros_like(frame_src[:, 0, :, :]),
                "cycle_similarity": torch.zeros_like(frame_src[:, 0, :, :]),
                "reconstruction": frame_warped,
                "flow": flow,
                "flow_low": flow_low,
                "corr_mask": corr_mask,
                "correlation": correlation,
            }
        vis_dict["feature0"] = feature0
        vis_dict["feature1"] = feature1
        return loss_dict, vis_dict

    def forward_features(self, frame_src: torch.Tensor, frame_dst: torch.Tensor):
        return self.unimatch.forward_features(frame_src, frame_dst)


def charbonnier_loss(delta, alpha=0.45, epsilon=1e-3):
    loss = torch.pow(torch.square(delta) + epsilon**2, alpha)
    return loss


def compute_smoothness_loss(flow, mask):
    """Local smoothness loss, as defined in equation (5) of the paper.

    The neighborhood here is defined as the 8-connected region around each pixel.
    """

    def get_loss(x, mask):
        # x: B x 2 x H x W
        # mask: B X N x H x W
        loss = charbonnier_loss(x)
        # B X N x H x W
        loss = loss.sum(dim=1, keepdim=True) * mask
        loss = torch.sum(loss, dim=(2, 3)) / (torch.sum(mask, dim=(2, 3)) + 1e-8)
        return loss.mean(1)

    flow_ucrop = flow[:, :, 1:, :]
    flow_dcrop = flow[:, :, :-1, :]
    flow_lcrop = flow[:, :, :, 1:]
    flow_rcrop = flow[:, :, :, :-1]

    flow_ulcrop = flow[:, :, 1:, 1:]
    flow_drcrop = flow[:, :, :-1, :-1]
    flow_dlcrop = flow[:, :, :-1, 1:]
    flow_urcrop = flow[:, :, 1:, :-1]
    smoothness_loss = 0.0
    smoothness_loss += get_loss(flow_ucrop - flow_dcrop, mask[:, :, 1:, :])
    smoothness_loss += get_loss(flow_lcrop - flow_rcrop, mask[:, :, :, 1:])
    smoothness_loss += get_loss(flow_ulcrop - flow_drcrop, mask[:, :, 1:, 1:])
    smoothness_loss += get_loss(flow_dlcrop - flow_urcrop, mask[:, :, :-1, 1:])
    smoothness_loss /= 4.0
    return smoothness_loss


def compute_cycle_consistency_loss(forward_flow, backward_flow, mask):
    cycle_diff = flow_warp(backward_flow, forward_flow) + forward_flow
    cycle_dist = cycle_diff.norm(dim=1) * mask
    cycle_loss = cycle_dist.sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-8)
    return cycle_loss


def compute_semantic_loss(dino_affinity, pred_prob, positive_threshold=0.3):
    """
    Apply cross entropy loss only on negative affinity
    Suppress the logits of negative affinity using label from dino
    dino_affinity: B x N x M
    pred_corr: B x N x M
    """
    aff_percentile = torch.quantile(
        dino_affinity, positive_threshold, dim=2, interpolation="higher", keepdim=True
    )
    dino_affinity_label = (dino_affinity >= aff_percentile).float()
    neg_prob = (pred_prob * (1 - dino_affinity_label)).sum(dim=2)
    loss = torch.log(neg_prob + 1e-8).mean(1)
    return loss


def get_eight_way_distance(coords):
    def _get_euclidean_distance(x, y):
        return torch.sqrt(torch.sum((x - y) ** 2, dim=1, keepdim=True) + 1e-8)

    # coords: B x 2 x H x W
    coords_ucrop = coords[:, :, 1:, :]
    coords_dcrop = coords[:, :, :-1, :]
    coords_lcrop = coords[:, :, :, 1:]
    coords_rcrop = coords[:, :, :, :-1]

    coords_ulcrop = coords[:, :, 1:, 1:]
    coords_drcrop = coords[:, :, :-1, :-1]
    coords_dlcrop = coords[:, :, :-1, 1:]
    coords_urcrop = coords[:, :, 1:, :-1]
    dist_ud = _get_euclidean_distance(coords_ucrop, coords_dcrop)
    dist_lr = _get_euclidean_distance(coords_lcrop, coords_rcrop)
    dist_uldr = _get_euclidean_distance(coords_ulcrop, coords_drcrop)
    dist_dlur = _get_euclidean_distance(coords_dlcrop, coords_urcrop)
    return dist_ud, dist_lr, dist_uldr, dist_dlur


def compute_distance_consistency_loss(flow, mask):
    """Local smoothness loss, as defined in equation (5) of the paper.

    The neighborhood here is defined as the 8-connected region around each pixel.
    """

    def get_loss(x, mask):
        # x: B x 2 x H x W
        # mask: B X N x H x W
        loss = charbonnier_loss(x)
        # B X N x H x W
        loss = loss.sum(dim=1, keepdim=True) * mask
        loss = torch.sum(loss, dim=(2, 3)) / (torch.sum(mask, dim=(2, 3)) + 1e-8)
        return loss.mean(1)

    b, _, h, w = flow.shape
    coords_src = coords_grid(b, h, w).to(flow.device)
    coords_dst = coords_src + flow

    dist_ud_src, dist_lr_src, dist_uldr_src, dist_dlur_src = get_eight_way_distance(coords_src)
    dist_ud_dst, dist_lr_dst, dist_uldr_dst, dist_dlur_dst = get_eight_way_distance(coords_dst)

    smoothness_loss = 0.0
    smoothness_loss += get_loss(dist_ud_src - dist_ud_dst, mask[:, :, 1:, :])
    smoothness_loss += get_loss(dist_lr_src - dist_lr_dst, mask[:, :, :, 1:])
    smoothness_loss += get_loss(dist_uldr_src - dist_uldr_dst, mask[:, :, 1:, 1:])
    smoothness_loss += get_loss(dist_dlur_src - dist_dlur_dst, mask[:, :, :-1, 1:])
    smoothness_loss /= 4.0
    return smoothness_loss


def huber_loss(
    tracks: torch.Tensor,
    target_points: torch.Tensor,
    occluded: torch.Tensor,
) -> torch.Tensor:
    """Huber loss for point trajectories."""
    # tracks: B, N, 2
    error = tracks - target_points
    # Huber loss with a threshold of 4 pixels
    distsqr = torch.sum(torch.square(error), dim=-1)
    dist = torch.sqrt(distsqr + 1e-12)  # add eps to prevent nan
    delta = 4.0
    loss_huber = torch.where(
        dist < delta,
        distsqr / 2,
        delta * (torch.abs(dist) - delta / 2),
    )
    loss_huber *= 1.0 - occluded

    loss_huber = torch.mean(loss_huber, dim=1)

    return loss_huber


def query_point_correlation(points, correlation):
    # points: B * N * 2 (x, y), [0, 1]
    # correlation: B * h * w * h * w
    B, N, _ = points.shape
    h, w = correlation.shape[1], correlation.shape[2]
    query_points_norm = points * 2 - 1  # B * N * 2, in range [-1,1]
    query_points_norm = rearrange(query_points_norm, "b n c -> (b n) 1 1 c")  # (B*N) * 1 * 1 * 2
    correlation = rearrange(correlation, "b h w h2 w2 -> b (h2 w2) h w")
    correlation = correlation.unsqueeze(1).expand(-1, N, -1, -1, -1)
    correlation = rearrange(correlation, "b n hw2 h w -> (b n) hw2 h w")
    # (B*N) * (h*w) * 1 * 1
    point_correlation = F.grid_sample(correlation, query_points_norm, align_corners=False)
    point_correlation = point_correlation.squeeze(-1).squeeze(-1)
    point_correlation = rearrange(point_correlation, "(b n) (h w) -> b n h w", b=B, h=h)
    return point_correlation


@torch.no_grad()
def extract_dino_feature(model, frame, return_h_w=False):
    """frame: B, C, H, W"""
    B = frame.shape[0]
    out = model.get_intermediate_layers(frame, n=1)[0]
    out = out[:, 1:, :]  # we discard the [CLS] token
    h, w = int(frame.shape[2] / model.patch_embed.patch_size), int(
        frame.shape[3] / model.patch_embed.patch_size
    )
    dim = out.shape[-1]
    out = out.reshape(B, -1, dim)
    if return_h_w:
        return out, h, w
    return out


@torch.no_grad()
def get_dino_corr_mask(
    model, frame_src, frame_dst, mask_ratio=0.9, binary=True, return_feat=False
):
    # frame_src: B x C x H x W
    # frame_dst: B x C x H x W
    # mask_ratio: ratio of pixels to be masked
    # return: B x h*w x h*w
    feat_1, h, w = extract_dino_feature(model, frame_src, return_h_w=True)
    feat_2 = extract_dino_feature(model, frame_dst)

    feat_1_norm = F.normalize(feat_1, dim=2, p=2)
    feat_2_norm = F.normalize(feat_2, dim=2, p=2)
    aff_raw = torch.einsum("bnc,bmc->bnm", [feat_1_norm, feat_2_norm])

    # bottom are masked
    if binary:
        if mask_ratio <= 0:
            # no corr mask
            corr_mask = None
        else:
            if aff_raw.dtype == torch.float16:
                aff_raw = aff_raw.float()
            aff_percentile = torch.quantile(aff_raw, mask_ratio, 2, keepdim=True)
            # True for masked
            corr_mask = aff_raw < aff_percentile
    else:
        # normalize to [0, 1] and -1 for masked
        # corr_mask = (aff_raw - aff_percentile) / (
        #     torch.max(aff_raw, dim=2, keepdim=True)[0] - aff_percentile
        # )
        # corr_mask[aff_raw < aff_percentile] = -1
        corr_mask = aff_raw
    if return_feat:
        B, _, C = feat_1.shape
        feat_1 = feat_1.view(B, h, w, C).permute(0, 3, 1, 2)
        feat_2 = feat_2.view(B, h, w, C).permute(0, 3, 1, 2)
        return corr_mask, feat_1, feat_2
    else:
        return corr_mask
