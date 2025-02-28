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
            rotate: bool = False,
            augment_image: bool = False,
    ) -> None:
        """

        :param dataset_dir_path: The path to the directory containing the dataset.
        :param len_override: Optional. Override the length of the dataset. If not provided, the length of the dataset
        :param resize: Resize the images and targets to 256x256. Defaults to False.
        :param normalize: Normalize the images. Defaults to False.
        :param rotate: Rotate the images and targets by a random multiple of 90 degrees. Defaults to False.
        :param augment_image: Augment the images. Defaults to False.
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
        self.__normalize = _transforms.Normalize(mean=MEAN, std=STD) if normalize else None
        self.__to_tensor = _transforms.PILToTensor()
        self.__resize = _transforms.Resize((256, 256)) if resize else None
        self.__rotate = rotate
        self.__augment_image = augment_image
        self.__augmentations = [
            _transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 1.0)),
            _transforms.ColorJitter(brightness=(0.5, 1.5)),
            _transforms.ColorJitter(
                contrast=(0.5, 1.5),
                saturation=(0.5, 1.5),
                hue=0.125,
            ),
            _nn.Identity(),
        ] if augment_image else None

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
        rotate = self.__rotate
        augment_image = self.__augment_image
        augmentations = self.__augmentations

        # Load, resize and convert the image and target to tensors
        image_path, target_path = image_target_paths[idx]
        image, target = _Image.open(image_path), _Image.open(target_path)

        # Resize the image and target to 256x256
        if resize is not None:
            image, target = resize(image), resize(target)

        image, target = to_tensor(image).float() / 255, to_tensor(target).float() // 255

        # Rotate the image and target by a random multiple of 90 degrees
        if rotate:
            k = _torch.randint(0, 4, (1,)).item()
            image = _torch.rot90(image, k=k, dims=(1, 2))
            target = _torch.rot90(target, k=k, dims=(1, 2))

        # Augment the image
        if augment_image:
            augmentation = _random.choice(augmentations)
            if _random.random() < 0.5:
                # Only apply the augmentation to the background
                image = target * image + (1 - target) * augmentation(image)
            else:
                image = augmentation(image)

        # Normalize the image
        if normalize is not None:
            image = normalize(image)

        return image, target


# --------------------------------------------
# Public Constants
# --------------------------------------------

MEAN = [0.4808, 0.4178, 0.5046]
STD = [0.2637, 0.2751, 0.2425]

# --------------------------------------------
# Private Constants
# --------------------------------------------

_IMAGES_DIR_NAME = 'image'
_TARGETS_DIR_NAME = 'mask'
