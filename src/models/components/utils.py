import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from PIL import Image

from src.datamodules.components.transforms import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    LAB_MEAN,
    LAB_STD,
)


def tensor2image(tensor: torch.Tensor, to_lab: bool = False):
    """Converts a tensor to an image.

    Args:
        tensor (torch.Tensor): Tensor to convert. Shape: (C, H, W) or (B, C, H, W).
        mean (list, optional): Mean of the dataset. Defaults to IMAGENET_DEFAULT_MEAN.
        std (list, optional): Standard deviation of the dataset. Defaults to IMAGENET_DEFAULT_STD.

    Returns:
        np.ndarray: Image array. Shape: (H, W, C) or (B, H, W, C).
    """
    if to_lab:
        mean = torch.tensor(LAB_MEAN, device=tensor.device)
        std = torch.tensor(LAB_STD, device=tensor.device)
    else:
        mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=tensor.device)
        std = torch.tensor(IMAGENET_DEFAULT_STD, device=tensor.device)
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    if len(tensor.shape) == 3:
        img_tensor = tensor[None, :, :, :]
    elif len(tensor.shape) == 4:
        img_tensor = tensor
    else:
        raise ValueError(f"Invalid tensor shape: {tensor.shape}")
    img_tensor = img_tensor * std + mean
    img_array = img_tensor.permute(0, 2, 3, 1).cpu().numpy()
    if to_lab:
        img_array = [cv2.cvtColor(img.astype(np.float32), cv2.COLOR_Lab2RGB) for img in img_array]
        img_array = np.stack(img_array)
    img_array = img_array.clip(0, 1)
    img_array = (img_array * 255).astype(np.uint8)
    if len(tensor.shape) == 3:
        img_array = img_array[0]
    return img_array


def fig2numpy(fig):
    """Converts a matplotlib figure to a numpy array.

    Args:
        fig (matplotlib.figure.Figure): Figure to convert.

    Returns:
        np.ndarray: Numpy array of the figure.
    """
    fig.canvas.draw()
    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data.copy()


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col : col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col : col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col : col + MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, "input flow must have three dimensions"
    assert flow_uv.shape[2] == 2, "input flow must have shape [H,W,2]"
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def vis_train(batch, pred, to_lab=False):
    imgs_src = tensor2image(batch["frame_src"], to_lab)
    imgs_dst = tensor2image(batch["frame_dst"], to_lab)
    # resize mask to match the size of imgs_src
    # cycle_similarity = torch.nn.functional.interpolate(
    #     pred["cycle_similarity"], size=imgs_src.shape[1:3], mode="nearest"
    # ).squeeze(1)
    cycle_similarity = pred["cycle_similarity"].squeeze(1)
    # selected_mask = torch.nn.functional.interpolate(
    #     pred["selected_mask"], size=imgs_src.shape[1:3], mode="nearest"
    # ).squeeze(1)
    selected_mask = pred["selected_mask"].squeeze(1)
    flow = pred["flow"][0].detach().cpu().numpy()
    flow_img = flow_to_image(flow.transpose(1, 2, 0))

    cycle_similarity = cycle_similarity.cpu().numpy()
    selected_mask = selected_mask.cpu().numpy()
    imgs_recon = tensor2image(pred["reconstruction"].detach(), to_lab)
    fig, axs = plt.subplots(1, 6, figsize=(16, 4), dpi=100)
    axs[0].imshow(imgs_src[0])
    axs[0].set_title("query")

    axs[1].imshow(imgs_dst[0])
    axs[1].set_title("reference")

    axs[2].imshow(imgs_src[0])
    axs[2].imshow(cycle_similarity[0], alpha=0.5)
    axs[2].set_title("masked query")

    axs[3].imshow(imgs_src[0], alpha=0.5)
    axs[3].imshow(selected_mask[0])
    axs[3].set_title("selected mask")

    axs[4].imshow(imgs_recon[0])
    axs[4].set_title("warped dst")

    axs[5].imshow(flow_img)
    axs[5].set_title("flow")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    fig_array = fig2numpy(fig)
    fig_img = Image.fromarray(fig_array)
    plt.close(fig)
    return fig_img


def vis_train_point(batch, vis_dict, to_lab=False):
    b_idx = 0
    query_point = batch["query_points"].cpu().numpy()[b_idx]
    target_point = batch["target_points"].cpu().numpy()[b_idx]
    pred_point = vis_dict["points"].detach().cpu().numpy()[b_idx]
    pred_visible = (vis_dict["occ_logits"].detach().sigmoid() < 0.5).cpu().numpy()[b_idx]
    frame_src = tensor2image(batch["frame_src"], to_lab=to_lab)[b_idx]
    frame_dst = tensor2image(batch["frame_dst"], to_lab=to_lab)[b_idx]
    visible = ~batch["occluded"].cpu().numpy()[b_idx]

    colors = cm.hsv(np.arange(query_point.shape[0]) / query_point.shape[0])
    fig, axs = plt.subplots(1, 3, figsize=(13, 5))
    axs[0].imshow(frame_src)
    axs[0].scatter(query_point[visible, 0], query_point[visible, 1], c=colors[visible], s=20)
    axs[0].scatter(
        query_point[~visible, 0], query_point[~visible, 1], c=colors[~visible], s=20, marker="x"
    )

    axs[1].imshow(frame_dst)
    axs[1].scatter(target_point[visible, 0], target_point[visible, 1], c=colors[visible], s=20)
    axs[2].imshow(frame_dst)
    axs[2].scatter(
        pred_point[pred_visible, 0], pred_point[pred_visible, 1], c=colors[pred_visible], s=20
    )
    axs[2].scatter(
        pred_point[~pred_visible, 0],
        pred_point[~pred_visible, 1],
        c=colors[~pred_visible],
        s=20,
        marker="x",
    )
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    fig_array = fig2numpy(fig)
    fig_img = Image.fromarray(fig_array)
    plt.close(fig)
    return fig_img


def vis_val(batch, similarity, tracks, to_lab=False):
    start_frame_idxs = batch["query_points"][:, :, 0].long()
    # N * 3 * H * W
    start_frame_per_point = batch["frames"][0, start_frame_idxs.squeeze(0), :, :, :]
    start_frame_per_point = tensor2image(start_frame_per_point, to_lab)
    # get the last frame where the point is visible
    last_visible_idx = (
        batch["occluded"][0].shape[1]
        - torch.argmax(torch.logical_not(batch["occluded"][0]).long().flip(1), dim=1)
        - 1
    )
    end_frame_per_point = batch["frames"][0, last_visible_idx, :, :, :]
    end_frame_per_point = tensor2image(end_frame_per_point, to_lab)

    figs = []
    for n in range(last_visible_idx.size(0)):
        start_frame = start_frame_per_point[n]
        end_frame = end_frame_per_point[n]
        # source_pt: (y, x)
        source_pt = batch["query_points"][0, n, 1:].float().cpu().numpy()
        source_pt = source_pt * np.array([start_frame.shape[0], start_frame.shape[1]])
        # source_pt: (x, y)
        target_pt = batch["target_points"][0, n, last_visible_idx[n]].cpu().numpy()
        target_pt = target_pt * np.array([end_frame.shape[1], end_frame.shape[0]])
        # pred_pt: (x, y)
        pred_pt = tracks[0, n, last_visible_idx[n]].cpu().numpy()
        pred_pt = pred_pt * np.array([end_frame.shape[1], end_frame.shape[0]])

        response = similarity[:, n, [-1]]
        response = torch.nn.functional.interpolate(
            response,
            size=(start_frame.shape[0], start_frame.shape[1]),
            mode="bilinear",
            align_corners=False,
        )

        response = response.squeeze().cpu().numpy()
        fig, axs = plt.subplots(1, 3, figsize=(9, 3), dpi=100)
        axs[0].imshow(start_frame)
        axs[0].scatter(source_pt[1], source_pt[0], c="r", s=50, marker="x")
        axs[0].axis("off")
        axs[0].set_title("Source")
        axs[1].imshow(end_frame)
        axs[1].imshow(response, cmap=cm.jet, alpha=0.5)
        axs[1].scatter(pred_pt[0], pred_pt[1], c="w", s=50, marker="x")
        axs[1].axis("off")
        axs[1].set_title("Response")
        axs[2].imshow(end_frame)
        axs[2].scatter(target_pt[0], target_pt[1], c="r", s=50, marker="x")
        axs[2].axis("off")
        axs[2].set_title("Target")
        plt.tight_layout()
        fig_array = fig2numpy(fig)
        figs.append(Image.fromarray(fig_array))
        plt.close(fig)
    return figs


def cosine_scheduler(
    base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule
