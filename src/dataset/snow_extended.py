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
            len_override: int = None,
            resize: bool = False,
            normalize: bool = False,
    ) -> None:
        """

        :param dataset_dir_path: The path to the directory containing the dataset.
        :param len_override: Optional. Override the length of the dataset. If not provided, the length of the dataset
        :param resize: Optional. Resize the images and targets to 256x256.
        :param normalize: Optional. Normalize the images.
        """
        images_dir_path = _os.path.join(dataset_dir_path, _IMAGES_DIR_NAME)
        targets_dir_path = _os.path.join(dataset_dir_path, _TARGETS_DIR_NAME)
        image_target_paths = tuple(
            tuple([_os.path.join(images_dir_path, file_name), _os.path.join(targets_dir_path, file_name)])
            for file_name in sorted(set(_os.listdir(targets_dir_path)).intersection(set(_os.listdir(images_dir_path))))
        )
        len_image_target_paths = len(image_target_paths)
        if len_override is not None and len_override < len_image_target_paths:
            image_target_paths = image_target_paths[:len_override]
            len_image_target_paths = len_override

        if len_image_target_paths == 0:
            raise ValueError('The dataset directory does not contain any images or targets.')

        self.__count_original = len_image_target_paths
        self.__count_total = len_image_target_paths * 4
        self.__image_target_paths = image_target_paths
        self.__normalize = _transforms.Normalize(mean=_MEAN, std=_STD) if normalize else _nn.Identity()
        self.__to_tensor = _transforms.PILToTensor()
        self.__resize = _transforms.Resize((256, 256)) if resize else _nn.Identity()

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        :return: The length of the dataset.
        """
        count_total = self.__count_total
        return count_total

    def __getitem__(self, idx) -> Tuple[_torch.Tensor, _torch.Tensor]:
        """
        Returns the image and target at the given index.

        :param idx: The index of the image and target.
        :return: A tuple containing the image and target.
        """
        image_target_paths = self.__image_target_paths
        count_original = self.__count_original
        normalize = self.__normalize
        to_tensor = self.__to_tensor
        resize = self.__resize

        if idx >= count_original:
            # Request for augmented image which is not in cache yet
            idx_original = idx % count_original
            # Load the original image and target
            image, target = self.__getitem__(idx_original)
            # Rotate the image and target
            image, target = self._rotate(image, target, k=idx // count_original)
        else:
            # Else, load, resize and convert the image and target to tensors
            image_path, target_path = image_target_paths[idx]
            image, target = _Image.open(image_path), _Image.open(target_path)
            image, target = resize(image), resize(target)
            image, target = to_tensor(image).float() / 255, to_tensor(target).float() // 255
            image = normalize(image)

        return image, target

    @staticmethod
    def _rotate(image: _Image, target: _Image, k: int) -> Tuple[_torch.Tensor, _torch.Tensor]:
        """
        Rotates the image and target by `k` 90 degrees.

        :param image: The image to rotate.
        :param target: The target to rotate.
        :param k: The number of times to rotate the image and target by 90 degrees.
        :return: A tuple containing the rotated image and target.
        """

        # Rotate both tensors by the same amount
        image = _torch.rot90(image, k=k, dims=(1, 2))
        target = _torch.rot90(target, k=k, dims=(1, 2))

        return image, target


# --------------------------------------------
# Private Constants
# --------------------------------------------


_IMAGES_DIR_NAME = 'image'
_TARGETS_DIR_NAME = 'mask'
_MEAN = [0.4808, 0.4178, 0.5046]
_STD = [0.2637, 0.2751, 0.2425]
