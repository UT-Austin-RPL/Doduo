import glob
import pickle

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.datamodules.components.transforms import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    LAB_MEAN,
    LAB_STD,
)


class TapVidTransform:
    def __init__(self, size=(224, 224), full_res=False):
        self.size = size
        self.full_res = full_res

    def __call__(self, x):
        h, w = x["video"].shape[1:3]
        frames = torch.from_numpy(x["video"]).permute(0, 3, 1, 2).float()
        frames = frames / 255.0
        frames = frames - torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
        frames = frames / torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)
        if self.full_res:
            # avoid OOM
            if w > 1000:
                h = 320
            else:
                h = 480
            w = int(1.0 * frames.shape[3] / frames.shape[2] * h)
            w = (w // 16) * 16
            size = (h, w)
        else:
            size = (self.size[0], self.size[1])
        frames = torch.nn.functional.interpolate(
            frames, size=size, mode="bilinear", align_corners=False
        )
        data = {
            "frames": frames,
        }
        # resize points
        data["points"] = x["points"]
        # data['points'][:, :, 0] = data['points'][:, :, 0] / w * size[0]
        # data['points'][:, :, 1] = data['points'][:, :, 1] / h * size[1]
        data["occluded"] = x["occluded"]
        return data

class TAPVidDAVISDataset(Dataset):
    def __init__(self, path, transform=None):
        with open(path, "rb") as f:
            self.data = pickle.load(f)
        self.video_names = list(self.data.keys())
        self.video_names.sort()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        data = self.data[video_name]
        # dict with keys:
        # points: N * T * 2
        # occluded: N * T
        # frames: T * 3 * H * W
        data = self.transform(data)

        target_occluded = data["occluded"]
        target_points = data["points"]

        valid = np.sum(~target_occluded, axis=1) > 0
        target_points = target_points[valid, :]
        target_occluded = target_occluded[valid, :]

        query_points = []
        for i in range(target_points.shape[0]):
            index = np.where(target_occluded[i] == 0)[0][0]
            x, y = target_points[i, index, 0], target_points[i, index, 1]
            query_points.append(np.array([index, y, x]))  # [t, y, x]
        query_points = np.stack(query_points, axis=0)
        return {
            "frames": data["frames"],  # T * C * H * W
            "query_points": query_points,  # N * 3 (t, y, x)
            "target_points": target_points,  # N * T * 2 (x, y)
            "occluded": target_occluded,  # N * T
            "video_name": video_name,
        }
