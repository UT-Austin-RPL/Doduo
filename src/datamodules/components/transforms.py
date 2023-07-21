import cv2
import numpy as np
import torch
from kornia import augmentation as K
from PIL import Image
from torch import nn
from torchvision.transforms import transforms

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
LAB_MEAN = [50, 0, 0]
LAB_STD = [50, 127, 127]


class SegTransform(nn.Module):
    """Basic transform for image and segmentation.

    Add random cropping
    """

    def __init__(self, base_size, crop_size, hflip_p=0.0, **kwargs):
        super().__init__()
        self.transform_img = transforms.ToTensor()
        self.transform_seg = lambda x: torch.from_numpy(np.array(x))
        self.joint_transform = K.AugmentationSequential(
            K.Resize(base_size),
            K.RandomCrop(crop_size),
            K.RandomHorizontalFlip(p=hflip_p),
            K.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            data_keys=["input", "mask"],  # Just to define the future input here.
            same_on_batch=False,
        )

    def forward(self, img, seg):
        img_t = self.transform_img(img).unsqueeze(0)
        seg_t = self.transform_seg(seg).unsqueeze(0)
        seg_t = seg_t.unsqueeze(0)
        seg_t = seg_t.float()
        img_t, seg_t = self.joint_transform(img_t, seg_t)
        return img_t.squeeze(0), seg_t.squeeze(0).squeeze(0)
