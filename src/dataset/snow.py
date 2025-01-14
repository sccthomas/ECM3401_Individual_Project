import os as _os
from typing import Tuple

import torch as _torch
import torchvision.transforms.v2 as _transforms
from PIL import Image as _Image
from torch.utils.data import Dataset as _Dataset


class SnowDataset(_Dataset):
    """
    A dataset class for the snow dataset.
    """

    def __init__(self, dataset_dir_path: str, len_override: int = None) -> None:
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

        self.__count = len(image_target_paths) if len_override is None else len_override
        self.__image_target_paths = image_target_paths
        self.__normalize = _transforms.Normalize(mean=_MEAN, std=_STD)
        self.__random_horizontal_flip = _transforms.RandomHorizontalFlip()
        self.__random_vertical_flip = _transforms.RandomVerticalFlip()
        self.__to_tensor = _transforms.PILToTensor()
        self.__resize = _transforms.Resize((256, 256))

        self.__cache = [None] * self.__count

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
        image_transformed, target_transformed = random_horizontal_flip(image, target)
        image_transformed, target_transformed = random_vertical_flip(image_transformed, target_transformed)

        return image_transformed, target_transformed


# --------------------------------------------
# Private Constants
# --------------------------------------------


_IMAGES_DIR_NAME = 'image'
_TARGETS_DIR_NAME = 'mask'
_MEAN = [0.4808, 0.4178, 0.5046]
_STD = [0.2637, 0.2751, 0.2425]
