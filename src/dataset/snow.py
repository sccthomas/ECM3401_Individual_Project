import os as _os
from typing import Tuple

import torch as _torch
import torch.nn as _nn
import torchvision.transforms.v2 as _transforms
from PIL import Image as _Image
from torch.utils.data import Dataset as _Dataset


class SnowDataset(_Dataset):
    """
    Dataset class for the Snow dataset. This dataset contains images of histopathological slides of human breast tissue
    and their corresponding masks. The images are RGB images of size 512x512 and the masks are binary images of size
    512x512. The dataset contains 20,000 images and their corresponding masks. The images and masks are stored in two
    separate directories. The images are stored in a directory named 'image' and the masks are stored in a directory
    named 'mask'. The images and masks are stored in the same order. The images and masks are named using the same
    file name.
    """

    def __init__(
            self,
            dataset_dir_path: str,
            cache: dict,
            len_override: int = None,
            resize: bool = False,
            normalize: bool = False,
    ) -> None:
        """

        :param dataset_dir_path: The path to the directory containing the dataset.
        :param len_override: Optional. Override the length of the dataset. If not provided, the length of the dataset
        :param resize: Optional. Resize the images and targets to 256x256.
        :param normalize: Optional. Normalize the images.
        :param rotate: Optional. Rotate the images and targets by a random multiple of 90 degrees.
        """
        images_dir_path = _os.path.join(dataset_dir_path, _IMAGES_DIR_NAME)
        targets_dir_path = _os.path.join(dataset_dir_path, _TARGETS_DIR_NAME)
        image_target_paths = tuple(
            tuple([_os.path.join(images_dir_path, file_name), _os.path.join(targets_dir_path, file_name)])
            for file_name in sorted(set(_os.listdir(targets_dir_path)).intersection(set(_os.listdir(images_dir_path))))
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
        self.__to_tensor = _transforms.PILToTensor()
        self.__resize = _transforms.Resize((256, 256)) if resize else _nn.Identity()
        self.__cache = cache

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        :return: The length of the dataset.
        """
        count = self.__count
        return count

    def __getitem__(self, idx) -> Tuple[_torch.Tensor, _torch.Tensor]:
        """
        Returns the image and target at the given index.

        :param idx: The index of the image and target.
        :return: A tuple containing the image and target.
        """
        image_target_paths = self.__image_target_paths
        normalize = self.__normalize
        to_tensor = self.__to_tensor
        resize = self.__resize
        cache = self.__cache

        image_path, target_path = image_target_paths[idx]
        key = (image_path, target_path)

        if key in cache:
            image, target = cache[key]
        else:
            # Load, resize and convert the image and target to tensors
            image, target = _Image.open(image_path), _Image.open(target_path)
            image, target = resize(image), resize(target)
            image, target = to_tensor(image).float() / 255, to_tensor(target).float() // 255
            image = normalize(image)
            cache[key] = (image, target)

        return image, target


# --------------------------------------------
# Private Constants
# --------------------------------------------


_IMAGES_DIR_NAME = 'image'
_TARGETS_DIR_NAME = 'mask'
_MEAN = [0.4808, 0.4178, 0.5046]
_STD = [0.2637, 0.2751, 0.2425]
