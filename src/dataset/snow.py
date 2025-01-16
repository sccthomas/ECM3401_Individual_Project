import os as _os
import random as _random
from typing import Tuple

import torch as _torch
import torch.nn as _nn
import torchvision.transforms.v2 as _transforms
from PIL import Image as _Image
from torch.utils.data import Dataset as _Dataset


class SnowDataset(_Dataset):
    """
    A dataset class for the snow dataset.
    """

    def __init__(
            self, dataset_dir_path: str, len_override: int = None, resize: bool = False, normalize: bool = False
    ) -> None:
        """

        :param dataset_dir_path: The path to the directory containing the dataset.
        :param len_override: Optional. Override the length of the dataset. If not provided, the length of the dataset
        """
        images_dir_path = _os.path.join(dataset_dir_path, _IMAGES_DIR_NAME)
        targets_dir_path = _os.path.join(dataset_dir_path, _TARGETS_DIR_NAME)
        image_target_paths = tuple(
            tuple([_os.path.join(images_dir_path, file_name), _os.path.join(targets_dir_path, file_name)])
            for file_name in set(_os.listdir(targets_dir_path)).intersection(set(_os.listdir(images_dir_path)))
        )
        len_image_target_paths = len(image_target_paths)
        if len_image_target_paths == 0:
            raise ValueError('The dataset directory does not contain any images or targets.')

        count = (
            len_image_target_paths
            if len_override is None
            else len_override
            if len_override <= len_image_target_paths
            else len_image_target_paths
        )
        self.__count = count
        self.__image_target_paths = image_target_paths
        self.__normalize = _transforms.Normalize(mean=_MEAN, std=_STD) if normalize else _nn.Identity()
        self.__random_horizontal_flip = _transforms.RandomHorizontalFlip()
        self.__random_vertical_flip = _transforms.RandomVerticalFlip()
        self.__to_tensor = _transforms.PILToTensor()
        self.__resize = _transforms.Resize((256, 256)) if resize else _nn.Identity()

        self.__cache = [None] * count

    def __len__(self) -> int:
        count = self.__count
        return count

    def __getitem__(self, idx) -> Tuple[_torch.Tensor, _torch.Tensor]:
        image_target_paths = self.__image_target_paths
        cache = self.__cache
        normalize = self.__normalize
        to_tensor = self.__to_tensor
        resize = self.__resize

        # If the image and target are already loaded, return them
        if cache[idx] is not None:
            image, target = cache[idx]
        else:
            # Else, load, resize and convert the image and target to tensors
            image_path, target_path = image_target_paths[idx]
            image, target = _Image.open(image_path), _Image.open(target_path)
            image, target = resize(image), resize(target)
            image, target = to_tensor(image).float() / 255, to_tensor(target).float() // 255
            image = normalize(image)
            cache[idx] = (image, target)

        # Transform the image and target
        image_transformed, target_transformed = self.__transform(image, target)

        return image_transformed, target_transformed

    def __transform(self, image: _Image, target: _Image) -> Tuple[_torch.Tensor, _torch.Tensor]:
        random_horizontal_flip = self.__random_horizontal_flip
        random_vertical_flip = self.__random_vertical_flip

        # Random horizontal and vertical flip
        k = _random.randint(0, 3)  # 0, 1, 2, or 3

        # Rotate both tensors by the same amount
        image_transformed = _torch.rot90(image, k=k, dims=(1, 2))
        target_transformed = _torch.rot90(target, k=k, dims=(1, 2))

        return image_transformed, target_transformed


# --------------------------------------------
# Private Constants
# --------------------------------------------


_IMAGES_DIR_NAME = 'image'
_TARGETS_DIR_NAME = 'mask'
_MEAN = [0.4808, 0.4178, 0.5046]
_STD = [0.2637, 0.2751, 0.2425]
