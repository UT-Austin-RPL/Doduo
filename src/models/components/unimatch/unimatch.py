import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNEncoder
from .geometry import coords_grid
from .matching import (
    global_correlation_softmax_prototype,
    local_correlation_softmax_prototype,
)
from .transformer import FeatureTransformer
from .utils import feature_add_position


class UniMatch(nn.Module):
    def __init__(
        self,
        num_scales=1,
        feature_channels=128,
        upsample_factor=8,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        bilinear_upsample=False,
        corr_fn="global",
    ):
        super().__init__()

        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.upsample_factor = upsample_factor
        self.bilinear_upsample = bilinear_upsample
        if corr_fn == "global":
            self.corr_fn = global_correlation_softmax_prototype
        elif corr_fn == "local":
            self.corr_fn = local_correlation_softmax_prototype
        else:
            raise NotImplementedError(f"Correlation function {corr_fn} not implemented")

        # CNN
        self.backbone = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales)

        # Transformer
        self.transformer = FeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=feature_channels,
            nhead=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
        )

        # convex upsampling similar to RAFT
        # concat feature0 and low res flow as input
        if not bilinear_upsample:
            self.upsampler = nn.Sequential(
                nn.Conv2d(2 + feature_channels, 256, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, upsample_factor**2 * 9, 1, 1, 0),
            )

    def extract_feature(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)  # [2B, C, H, W]
        features = self.backbone(concat)  # list of [2B, C, H, W], resolution from high to low

        # reverse: resolution from low to high
        features = features[::-1]

        feature0, feature1 = [], []

        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, 2, 0)  # tuple
            feature0.append(chunks[0])
            feature1.append(chunks[1])

        return feature0, feature1

    def correlate_feature(self, feature0, feature1, attn_splits=2, attn_type="swin"):
        feature0, feature1 = feature_add_position(
            feature0, feature1, attn_splits, self.feature_channels
        )
        feature0, feature1 = self.transformer(
            feature0,
            feature1,
            attn_type=attn_type,
            attn_num_splits=attn_splits,
        )
        b, c, h, w = feature0.shape
        feature0 = feature0.view(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
        feature1 = feature1.view(b, c, -1)  # [B, C, H*W]
        correlation = torch.matmul(feature0, feature1).view(b, h, w, h, w) / (
            c**0.5
        )  # [B, H, W, H, W]
        correlation = correlation.view(b, h * w, h * w)  # [B, H*W, H*W]
        return correlation

    def forward(
        self,
        img0,
        img1,
        attn_type="swin",
        attn_splits=2,
        return_feature=False,
        bidirectional=False,
        cycle_consistency=False,
        corr_mask=None,
    ):
        # list of features, resolution low to high
        feature0_list, feature1_list = self.extract_feature(img0, img1)  # list of features
        assert self.num_scales == 1  # multi-scale depth model is not supported yet
        scale_idx = 0
        feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]

        if cycle_consistency:
            # get both directions of features
            feature0, feature1 = torch.cat((feature0, feature1), dim=0), torch.cat(
                (feature1, feature0), dim=0
            )

        # add position to features
        feature0, feature1 = feature_add_position(
            feature0, feature1, attn_splits, self.feature_channels
        )

        # Transformer
        feature0, feature1 = self.transformer(
            feature0,
            feature1,
            attn_type=attn_type,
            attn_num_splits=attn_splits,
        )
        b, c, h, w = feature0.shape
        # downsampled_img0 = F.interpolate(img0, size=(h, w), mode="bilinear", align_corners=False)
        flow_coords = coords_grid(b, h, w).to(feature0.device)  # [B, 2, H, W]
        # values = torch.cat((downsampled_img0, flow_coords), dim=1)  # [B, 5, H, W]
        # correlation and softmax
        query_results, correlation = self.corr_fn(
            feature0, feature1, flow_coords, pred_bidir_flow=bidirectional, corr_mask=corr_mask
        )
        if bidirectional:
            flow_coords = torch.cat((flow_coords, flow_coords), dim=0)
            up_feature = torch.cat((feature0, feature1), dim=0)
        else:
            up_feature = feature0
        flow = query_results - flow_coords
        flow_up = self.upsample_flow(flow, up_feature, bilinear=self.bilinear_upsample)
        if return_feature:
            return flow_up, flow, correlation, feature0, feature1
        else:
            return flow_up, flow, correlation

    def forward_features(
        self,
        img0,
        img1,
        attn_type="swin",
        attn_splits=2,
    ):

        feature0_list, feature1_list = self.extract_feature(img0, img1)  # list of features
        assert self.num_scales == 1  # multi-scale depth model is not supported yet
        scale_idx = 0
        feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]
        # add position to features
        feature0, feature1 = feature_add_position(
            feature0, feature1, attn_splits, self.feature_channels
        )

        # Transformer
        feature0, feature1 = self.transformer(
            feature0,
            feature1,
            attn_type=attn_type,
            attn_num_splits=attn_splits,
        )
        return feature0, feature1

    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=8, is_depth=False):
        if bilinear:
            multiplier = 1 if is_depth else upsample_factor
            up_flow = (
                F.interpolate(
                    flow, scale_factor=upsample_factor, mode="bilinear", align_corners=False
                )
                * multiplier
            )
        else:
            concat = torch.cat((flow, feature), dim=1)
            mask = self.upsampler(concat)
            up_flow = upsample_flow_with_mask(
                flow, mask, upsample_factor=self.upsample_factor, is_depth=is_depth
            )
        return up_flow


def upsample_flow_with_mask(flow, up_mask, upsample_factor, is_depth=False):
    # convex upsampling following raft

    mask = up_mask
    b, flow_channel, h, w = flow.shape
    mask = mask.view(b, 1, 9, upsample_factor, upsample_factor, h, w)  # [B, 1, 9, K, K, H, W]
    mask = torch.softmax(mask, dim=2)

    multiplier = 1 if is_depth else upsample_factor
    up_flow = F.unfold(multiplier * flow, [3, 3], padding=1)
    up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h, w)  # [B, 2, 9, 1, 1, H, W]

    up_flow = torch.sum(mask * up_flow, dim=2)  # [B, 2, K, K, H, W]
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
    up_flow = up_flow.reshape(
        b, flow_channel, upsample_factor * h, upsample_factor * w
    )  # [B, 2, K*H, K*W]

    return up_flow
