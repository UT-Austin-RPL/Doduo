from typing import Iterable, Mapping, Tuple, Union

import numpy as np
import torch
from einops import rearrange


def compute_tapvid_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
) -> Mapping[str, np.ndarray]:
    """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.)

    See the TAP-Vid paper for details on the metric computation.  All inputs are
    given in raster coordinates.  The first three arguments should be the direct
    outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
    The paper metrics assume these are scaled relative to 256x256 images.
    pred_occluded and pred_tracks are your algorithm's predictions.

    This function takes a batch of inputs, and computes metrics separately for
    each video.  The metrics for the full benchmark are a simple mean of the
    metrics across the full set of videos.  These numbers are between 0 and 1,
    but the paper multiplies them by 100 to ease reading.

    Args:
       query_points: The query points, an in the format [t, y, x].  Its size is
         [b, n, 3], where b is the batch size and n is the number of queries
       gt_occluded: A boolean array of shape [b, n, t], where t is the number
         of frames.  True indicates that the point is occluded.
       gt_tracks: The target points, of shape [b, n, t, 2].  Each point is
         in the format [x, y]
       pred_occluded: A boolean array of predicted occlusions, in the same
         format as gt_occluded.
       pred_tracks: An array of track predictions from your algorithm, in the
         same format as gt_tracks.
       query_mode: Either 'first' or 'strided', depending on how queries are
         sampled.  If 'first', we assume the prior knowledge that all points
         before the query point are occluded, and these are removed from the
         evaluation.

    Returns:
        A dict with the following keys:

        occlusion_accuracy: Accuracy at predicting occlusion.
        pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
          predicted to be within the given pixel threshold, ignoring occlusion
          prediction.
        jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
          threshold
        average_pts_within_thresh: average across pts_within_{x}
        average_jaccard: average across jaccard_{x}
    """

    metrics = {}

    # Don't evaluate the query point.  Numpy doesn't have one_hot, so we
    # replicate it by indexing into an identity matrix.
    one_hot_eye = np.eye(gt_tracks.shape[2])
    query_frame = query_points[..., 0]
    query_frame = np.round(query_frame).astype(np.int32)
    evaluation_points = one_hot_eye[query_frame] == 0

    # If we're using the first point on the track as a query, don't evaluate the
    # other points.
    if query_mode == "first":
        for i in range(gt_occluded.shape[0]):
            index = np.where(gt_occluded[i] == 0)[0][0]
            evaluation_points[i, :index] = False
    elif query_mode != "strided":
        raise ValueError("Unknown query mode " + query_mode)

    # Occlusion accuracy is simply how often the predicted occlusion equals the
    # ground truth.
    occ_acc = np.sum(
        np.equal(pred_occluded, gt_occluded) & evaluation_points,
        axis=(1, 2),
    ) / np.sum(evaluation_points)
    metrics["occlusion_accuracy"] = occ_acc

    # Next, convert the predictions and ground truth positions into pixel
    # coordinates.
    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []
    avg_distance = np.sqrt(
        np.sum(
            np.square(pred_tracks - gt_tracks),
            axis=-1,
        )
    ) * (1 - gt_occluded)
    avg_distance = np.sum(avg_distance) / np.sum(1 - gt_occluded)
    metrics["avg_distance"] = np.array([avg_distance])
    for thresh in [1, 2, 4, 8, 16]:
        # True positives are points that are within the threshold and where both
        # the prediction and the ground truth are listed as visible.
        within_dist = np.sum(
            np.square(pred_tracks - gt_tracks),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)

        # Compute the frac_within_threshold, which is the fraction of points
        # within the threshold among points that are visible in the ground truth,
        # ignoring whether they're predicted to be visible.
        count_correct = np.sum(
            is_correct & evaluation_points,
            axis=(1, 2),
        )
        count_visible_points = np.sum(visible & evaluation_points, axis=(1, 2))
        frac_correct = count_correct / count_visible_points
        metrics["pts_within_" + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)

        true_positives = np.sum(is_correct & pred_visible & evaluation_points, axis=(1, 2))

        # The denominator of the jaccard metric is the true positives plus
        # false positives plus false negatives.  However, note that true positives
        # plus false negatives is simply the number of points in the ground truth
        # which is easier to compute than trying to compute all three quantities.
        # Thus we just add the number of points in the ground truth to the number
        # of false positives.
        #
        # False positives are simply points that are predicted to be visible,
        # but the ground truth is not visible or too far from the prediction.
        gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
        jaccard = true_positives / (gt_positives + false_positives)
        metrics["jaccard_" + str(thresh)] = jaccard
        all_jaccard.append(jaccard)
    metrics["average_jaccard"] = np.mean(
        np.stack(all_jaccard, axis=1),
        axis=1,
    )
    metrics["average_pts_within_thresh"] = np.mean(
        np.stack(all_frac_within, axis=1),
        axis=1,
    )
    return metrics


def get_track_by_feat(feats, query_points):
    # feats: B * T * C * H * W, normalized
    # query_points: B * N * 3, (t, y, x)
    # get query features
    query_points_norm = query_points.float()[:, :, 1:] * 2 - 1
    # B * N
    start_frame_idxs = query_points[:, :, 0].long()
    # B * N * c * h * w
    start_frame_feats = feats[torch.arange(feats.size(0)), start_frame_idxs, :, :, :]
    # (B * N) * c * h * w
    start_frame_feats = rearrange(start_frame_feats, "b n c h w -> (b n) c h w")
    # (B * N) * 1 * 1 * 2
    query_points_norm = rearrange(query_points_norm, "b n c -> (b n) 1 1 c")
    # flip last dimension to match the coordinate system of grid_sample
    query_feats = torch.nn.functional.grid_sample(
        start_frame_feats, query_points_norm.flip(-1), align_corners=False
    )
    # B * N * c
    query_feats = (
        rearrange(query_feats, "(b n) c h w -> b n c h w", b=feats.size(0)).squeeze(-1).squeeze(-1)
    )

    similarity = torch.einsum("bnc,btchw->bnthw", query_feats, feats)
    # get 2D argmax
    match_idx = torch.argmax(rearrange(similarity, "b n t h w -> b n t (h w)"), dim=-1)
    match_h = torch.div(
        match_idx, similarity.shape[-1], rounding_mode="floor"
    )  # match_idx // similarity.shape[-1]
    match_w = match_idx % similarity.shape[-1]
    match_h = match_h.float() / similarity.shape[-2]
    match_w = match_w.float() / similarity.shape[-1]
    # B, N, T, 2; x, y
    match_wh = torch.stack([match_w, match_h], dim=-1)
    return match_wh, similarity

    # def val_step(self, batch: Any):
    #     # val step
    #     # feature extraction
    #     frames = batch["frames"]
    #     query_points = batch["query_points"]  # B * N * 3, (t, y, x)
    #     B, N = query_points.shape[:2]
    #     _, T, _, H, W = frames.shape
    #     eval_visible_thresh = self.hparams.eval_visible_thresh * np.sqrt(H * W)
    #     # B * N
    #     start_frame_idxs = query_points[:, :, 0].long()
    #     start_frames = frames[torch.arange(B), start_frame_idxs]  # B * N * C * H * W
    #     start_frames = rearrange(start_frames, "b n c h w -> (b n) c h w")

    #     query_points_norm = query_points.float()[:, :, 1:] * 2 - 1
    #     # (B * N) * 1 * 1 * 2
    #     query_points_norm = rearrange(query_points_norm, "b n c -> (b n) 1 1 c")

    #     match_results = torch.zeros(B, N, T, 2).to(self.device)
    #     pred_occ = torch.zeros(B, N, T).to(self.device)
    #     sim_maps = []
    #     for t in range(T):
    #         target_frames = frames[:, [t]].repeat(1, N, 1, 1, 1)  # B * N * C * H * W
    #         target_frames = rearrange(target_frames, "b n c h w -> (b n) c h w")
    #         orig_batch_size = target_frames.shape[0]
    #         # (B * N) * C * H * W
    #         _, vis_dict = self.net(
    #             {"frame_src": start_frames, "frame_dst": target_frames},
    #             return_feature=True,
    #             cycle_consistency=True,
    #         )
    #         # (B * N) * 2 * H * W
    #         flow = vis_dict["flow"][:orig_batch_size]
    #         backward_flow = vis_dict["flow"][orig_batch_size:]
    #         # (B * N) * 1 * H * W
    #         cycle_dist = (flow_warp(backward_flow, flow) + flow).norm(dim=1, p=2, keepdim=True)
    #         query_cycle_dist = torch.nn.functional.grid_sample(
    #             cycle_dist, query_points_norm.flip(-1), align_corners=False
    #         )
    #         # B * N * 1
    #         query_cycle_dist = (
    #             rearrange(query_cycle_dist, "(b n) c h w -> b n c h w", b=B)
    #             .squeeze(-1)
    #             .squeeze(-1)
    #         )
    #         pred_occ[:, :, t] = query_cycle_dist.squeeze(-1) > eval_visible_thresh

    #         # get similarity map
    #         start_frame_feats, target_frames_feats = (
    #             vis_dict["feature0"][:orig_batch_size],
    #             vis_dict["feature1"][:orig_batch_size],
    #         )
    #         query_feats = torch.nn.functional.grid_sample(
    #             start_frame_feats, query_points_norm.flip(-1), align_corners=False
    #         )
    #         # B * N * c
    #         query_feats = (
    #             rearrange(query_feats, "(b n) c h w -> b n c h w", b=B).squeeze(-1).squeeze(-1)
    #         )
    #         target_frames_feats = rearrange(target_frames_feats, "(b n) c h w -> b n c h w", b=B)
    #         sim_map = torch.einsum("bnc,bnchw->bnhw", query_feats, target_frames_feats) / query_feats.size(-1) **0.5
    #         # extract matching from similarity map
    #         match_idx = torch.argmax(rearrange(sim_map, "b n h w -> b n (h w)"), dim=-1)
    #         match_h = torch.div(
    #             match_idx, sim_map.shape[-1], rounding_mode="floor"
    #         )  # match_idx // similarity.shape[-1]
    #         match_w = match_idx % sim_map.shape[-1]
    #         # B * N * 2
    #         match_coord = torch.stack([match_w, match_h], dim=-1).float()
    #         # B * 2 * h * w
    #         coords = coords_grid(sim_map.size(0), sim_map.size(2), sim_map.size(3), device=sim_map.device)
    #         # B * N * h * w
    #         dist_to_match = (coords.unsqueeze(1) - match_coord.unsqueeze(-1).unsqueeze(-1)).norm(dim=2, p=2)
    #         neighbor_mask = dist_to_match < self.hparams.eval_neighbor_thresh
    #         # set out of neighbor to -inf
    #         sim_map[~neighbor_mask] = -1e10
    #         # B * N * (h * w)
    #         prob = torch.nn.functional.softmax(sim_map.flatten(2), dim=-1)
    #         # B * N * 2
    #         query_results = torch.einsum("bnm,bdm->bnd", prob, coords.view(B, 2, -1))
    #         # normalize
    #         query_results = query_results / torch.tensor([sim_map.size(-2), sim_map.size(-1)]).to(self.device)
    #         match_results[:, :, t] = query_results
    #         sim_maps.append(sim_map)
    #     similarity = torch.stack(sim_maps, dim=2)  # B * N * T * h * w
    #     return match_results, similarity, pred_occ
