import os as _os
from typing import Tuple
from PIL import Image as _Image
import torch as _torch
from torch.utils.data import Dataset as _Dataset
from torchvision.transforms.v2 import ToTensor, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter


# --------------------------------------------
# Snow Dataset: Supervised Learning Classes
# --------------------------------------------


class SnowDataset(_Dataset):
    def __init__(self) -> None:
        dataset_dir_path = _os.path.join(_os.path.dirname(_os.getcwd()), _DIR_NAME)
        images_dir_path = _os.path.join(dataset_dir_path, _IMAGES_DIR_NAME)
        targets_dir_path = _os.path.join(dataset_dir_path, _TARGETS_DIR_NAME)
        image_target_paths = tuple(
            tuple([_os.path.join(images_dir_path, file_name), _os.path.join(targets_dir_path, file_name)])
            for file_name in set(_os.listdir(targets_dir_path)).intersection(set(_os.listdir(images_dir_path)))
        )

        self.__image_target_paths = image_target_paths
        self.__count = len(image_target_paths)
        self._to_tensor = ToTensor()

    def __len__(self) -> int:
        return self.__count

    def __getitem__(self, idx) -> Tuple[_torch.Tensor, _torch.Tensor]:
        image_target_paths = self.__image_target_paths

        image_path, target_path = image_target_paths[idx]

        image = _Image.open(image_path).convert('RGB')
        target = _Image.open(target_path).convert('L')

        image, target = self.transform(image, target)

        return image, target

    def transform(self, image: _Image, target: _Image) -> Tuple[_torch.Tensor, _torch.Tensor]:
        to_tensor = self._to_tensor

        image = to_tensor(image)
        target = to_tensor(target).float()
        return image, target


class SnowDatasetAugmented(SnowDataset):
    def __init__(self) -> None:
        super().__init__()
        self.__random_horizontal_flip = RandomHorizontalFlip()
        self.__random_vertical_flip = RandomVerticalFlip()
        self.__color_jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)

    def transform(self, image: _Image, target: _Image) -> Tuple[_torch.Tensor, _torch.Tensor]:
        random_horizontal_flip = self.__random_horizontal_flip
        random_vertical_flip = self.__random_vertical_flip
        color_jitter = self.__color_jitter
        to_tensor = self._to_tensor

        image, target = random_horizontal_flip(image, target)
        image, target = random_vertical_flip(image, target)
        image = color_jitter(image)

        image = to_tensor(image)
        target = to_tensor(target).float()

        return image, target


# --------------------------------------------
# Snow Dataset: Self-Supervised Learning Classes
# --------------------------------------------


class SnowDatasetContrastiveLearning(SnowDataset):
    def __init__(self) -> None:
        super().__init__()

    def transform(self, image: _torch.Tensor, target: _torch.Tensor) -> Tuple[_torch.Tensor, _torch.Tensor]:
        return image, target


class SnowDatasetContextPrediction(SnowDataset):
    def __init__(self) -> None:
        super().__init__()

    def transform(self, image: _torch.Tensor, target: _torch.Tensor) -> Tuple[_torch.Tensor, _torch.Tensor]:
        return image, target


class SnowDatasetMaskRegionPrediction(SnowDataset):
    def __init__(self) -> None:
        super().__init__()

    def transform(self, image: _torch.Tensor, target: _torch.Tensor) -> Tuple[_torch.Tensor, _torch.Tensor]:
        return image, target


# --------------------------------------------
# Private Constants
# --------------------------------------------


_DIR_NAME = 'snow_dataset'
_IMAGES_DIR_NAME = 'image'
_TARGETS_DIR_NAME = 'mask'
