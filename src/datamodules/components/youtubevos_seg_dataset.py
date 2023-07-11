import glob
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class YoutubeVOSSegDatasetV0(Dataset):
    """Basic Ego4d Frame dataset.

    Randomly select start frame and select near frame as end frame.
    """

    def __init__(
        self,
        data_root,
        max_interval,
        seg_dir="Annotations",
        max_class=8,
        transform=None,
        remove_bg=True,
        **kwargs,
    ):
        self.data_root = data_root
        if isinstance(max_interval, int):
            self.interval = (0, max_interval)
        else:
            self.interval = max_interval
        self.max_class = max_class
        self.seg_dir = seg_dir
        self.remove_bg = remove_bg

        data_paths = glob.glob(f"{data_root}/{seg_dir}/*/*.png")
        data_paths.sort()

        # count the number of frames on the left and right of each frame
        data_counter = {}
        self.available_range = {}
        for path in data_paths:
            video_name = path.split("/")[-2]
            if video_name not in data_counter:
                data_counter[video_name] = 0
            self.available_range[path] = [data_counter[video_name], 0]
            data_counter[video_name] += 1

        self.data_paths = []
        for path in data_paths:
            video_name = path.split("/")[-2]
            if data_counter[video_name] == 1:
                continue
            self.available_range[path][1] = (
                data_counter[video_name] - 1 - self.available_range[path][0]
            )
            self.data_paths.append(path)
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        while True:
            try:
                data_path = self.data_paths[idx]
                available_range = self.available_range[data_path]
                min_interval, max_interval = self.interval
                end_interval_0 = range(
                    max(idx - available_range[0], idx - max_interval),
                    max(idx - available_range[0], idx - min_interval),
                )
                end_interval_1 = range(
                    min(idx + available_range[1], idx + min_interval),
                    min(idx + available_range[1], idx + max_interval),
                )
                if len(end_interval_0) + len(end_interval_1) == 0:
                    end_interval = list(range(idx - available_range[0], idx + available_range[1]))
                else:
                    end_interval = list(end_interval_0) + list(end_interval_1)
                end_frame = np.random.choice(end_interval)

                start_seg_path = self.data_paths[idx]
                end_seg_path = self.data_paths[end_frame]
                start_path = start_seg_path.replace(self.seg_dir, "JPEGImages").replace(
                    ".png", ".jpg"
                )
                end_path = end_seg_path.replace(self.seg_dir, "JPEGImages").replace(".png", ".jpg")

                start_seg = Image.open(start_seg_path)
                end_seg = Image.open(end_seg_path)
                start_frame = Image.open(start_path)
                end_frame = Image.open(end_path)
                break
            except Exception as e:
                print(idx, e)
                idx = np.random.randint(0, len(self.data_paths))

        start_frame_name = "_".join(start_path.split("/")[-2:]).split(".")[0]
        end_frame_name = "_".join(end_path.split("/")[-2:]).split(".")[0]
        start_frame, start_seg = self.transform(start_frame, start_seg)
        end_frame, end_seg = self.transform(end_frame, end_seg)
        # seg to one-hot
        start_seg[start_seg >= self.max_class] = 0
        start_seg = start_seg.long().squeeze(0)
        start_seg = F.one_hot(start_seg, num_classes=self.max_class).permute(2, 0, 1)

        end_seg[end_seg >= self.max_class] = 0
        end_seg = end_seg.long()
        end_seg = F.one_hot(end_seg, num_classes=self.max_class).permute(2, 0, 1)

        if self.remove_bg:
            start_seg[0] = 0  # remove background
            end_seg[0] = 0  # remove background
        return {
            "frame_src": start_frame,  # C * H * W
            "frame_dst": end_frame,  # C * H * W
            "seg_src": start_seg,  # N * H * W
            "seg_dst": end_seg,  # N * H * W
            "frame_src_name": start_frame_name,
            "frame_dst_name": end_frame_name,
        }
