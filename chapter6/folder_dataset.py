from __future__ import absolute_import, division, print_function
import random
import numpy as np
from PIL import Image  # using pillow-simd for increased speed
import os
from os.path import join

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def load_image(filename):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(filename, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class FolderDataset(Dataset):
    """Superclass for monocular dataloaders"""

    def __init__(self,
                 data_path,
                 height,
                 width,
                 neighbor_ids,
                 start_index,
                 end_index,
                 is_train):
        """图像数据集

        Args:
            data_path (str): 图像文件夹 \\
            height (int): 输入图像的高 \\
            width (int): 输入图像的宽 \\
            neighbor_ids (list): 参与无监督学习的邻近帧，默认为三帧 \\
            start_index (int): 第一幅参与训练图像的编号 \\
            end_index (int): 最后一幅参与训练图像的编号 \\
            is_train (bool): 是否为训练数据集，用于开关数据增广 \\
        """
        super(FolderDataset, self).__init__()

        self.data_path = data_path
        self.filenames = sorted(os.listdir(data_path))
        self.height = height
        self.width = width
        self.interp = Image.LANCZOS
        self.is_train = is_train
        self.frame_idxs = neighbor_ids
        self.to_tensor = transforms.ToTensor()
        self.filenames = self.filenames[start_index:end_index]

        # GoPro Intrinsic
        self.K = np.array([[325.27862549 / 640, 0, 322.00147695 / 640, 0],
                           [0, 398.56437683 / 480, 243.16521174 / 480, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        # Need to specify augmentations differently in pytorch 1.0 compared with 0.4
        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)
        self.resize = transforms.Resize(
            (self.height, self.width), interpolation=self.interp)
        self.flag = np.zeros(self.__len__(), dtype=np.int64)

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            if "color" in k:
                color, img_index, scale = k
                inputs[(color, img_index, 0)] = self.resize(
                    inputs[(color, img_index, - 1)])

        # ('color', 0, 0)
        # ('color', -1, 0)
        # ('color', 1, 0)

        for k in list(inputs):
            if "color" in k:
                f = inputs[k]
                color, img_index, scale = k
                inputs[(color, img_index, scale)] = self.to_tensor(f)
                if scale == 0:
                    inputs[(color + "_aug", img_index, scale)
                           ] = self.to_tensor(color_aug(f))

        # ('color_aug', 0, 0)
        # ('color_aug', -1, 0)
        # ('color_aug', 1, 0)

        return inputs

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5

        for i in self.frame_idxs:
            try:
                filename = self.filenames[index+i]
            except:
                filename = self.filenames[index]

            inputs[("color", i, -1)] = load_image(join(self.data_path, filename))

        # adjusting intrinsics to match each scale in the pyramid
        K = self.K.copy()
        K[0, :] *= self.width
        K[1, :] *= self.height
        inv_K = np.linalg.pinv(K)

        inputs[("K")] = torch.from_numpy(K)
        inputs[("inv_K")] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)
        # ('color', 0, -1)
        # ('color', -1, -1)
        # ('color', 1, -1)
        # "K"
        # "inv_K"
        inputs = self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]

        return inputs
