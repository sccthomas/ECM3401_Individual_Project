import abc
import os as _os
from typing import Tuple
from PIL import Image as _Image
import torch as _torch
from torch.utils.data import Dataset as _Dataset
from torchvision.transforms.v2 import ToTensor, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter

# --------------------------------------------
# Snow Dataset
# --------------------------------------------


class SnowDataset(abc.ABC):
    def __init__(self) -> None:
        dataset_dir_path = _os.path.join(_os.path.dirname(_os.getcwd()), _DIR_NAME)
        images_dir_path = _os.path.join(dataset_dir_path, _IMAGES_DIR_NAME)
        file_names = _os.listdir(images_dir_path)

        self.__dataset_dir_path = dataset_dir_path
        self.__images_dir_path = images_dir_path
        self._file_names = file_names

    @property
    def dataset_dir_path(self) -> str:
        return self.__dataset_dir_path

    @property
    def image_dir_path(self) -> str:
        return self.__images_dir_path

    def get_image(self, idx: int) -> _Image:
        images_dir_path = self.image_dir_path
        file_names = self._file_names
        file_name = file_names[idx]
        image = _Image.open(_os.path.join(images_dir_path, file_name)).convert('RGB')

        return image

    @abc.abstractmethod
    def get_target(self, idx: int) -> _Image:
        """

        :param idx:
        :return:
        """


# --------------------------------------------
# Snow Dataset: Supervised Learning Classes
# --------------------------------------------


class SnowDatasetSupervisedLearning(SnowDataset, _Dataset):
    def __init__(self):
        super().__init__()
        targets_dir_path = _os.path.join(self.dataset_dir_path, _TARGETS_DIR_NAME)
        file_names = list(set(self._file_names).intersection(set(_os.listdir(targets_dir_path))))

        self.__targets_dir_path = targets_dir_path
        self._file_names = file_names
        self._file_count = len(file_names)
        self._to_tensor = ToTensor()

    def get_target(self, idx: int) -> _Image:
        targets_dir_path = self.__targets_dir_path
        file_names = self._file_names
        file_name = file_names[idx]
        target = _Image.open(_os.path.join(targets_dir_path, file_name)).convert('L')

        return target

    def _transform(self, image: _Image, target: _Image) -> Tuple[_torch.Tensor, _torch.Tensor]:
        to_tensor = self._to_tensor

        image = to_tensor(image)
        target = to_tensor(target).float()
        return image, target

    def __len__(self) -> int:
        file_count = self._file_count
        return file_count

    def __getitem__(self, idx) -> Tuple[_torch.Tensor, _torch.Tensor]:
        image = self.get_image(idx)
        target = self.get_target(idx)

        image, target = self._transform(image, target)

        return image, target


class SnowDatasetSupervisedLearningAugmented(SnowDatasetSupervisedLearning):
    def __init__(self):
        super().__init__()
        self.__random_horizontal_flip = RandomHorizontalFlip()
        self.__random_vertical_flip = RandomVerticalFlip()
        self.__color_jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)

    def _transform(self, image: _Image, target: _Image) -> Tuple[_torch.Tensor, _torch.Tensor]:
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


class SnowDatasetSelfSupervisedLearning(SnowDataset, _Dataset):
    def __init__(self):
        super().__init__()
        self._file_count = len(self._file_names)
        self._to_tensor = ToTensor()


    def get_target(self, idx: int) -> _Image:
        to_tensor = self._to_tensor

        image = to_tensor(self.get_image(idx))




    def _transform(self, image: _Image, target: _Image) -> Tuple[_torch.Tensor, _torch.Tensor]:
        to_tensor = self._to_tensor

        image = to_tensor(image)
        target = to_tensor(target).float()

        return image, target

    def __len__(self) -> int:
        file_count = self._file_count
        return file_count

    def __getitem__(self, idx) -> Tuple[_torch.Tensor, _torch.Tensor]:
        image = self.get_image(idx)
        target = self.get_target(idx)

        image, target = self._transform(image, target)

        return image, target


class SnowDatasetContrastiveLearning(SnowDatasetSelfSupervisedLearning):
    def __init__(self) -> None:
        super().__init__()

    def transform(self, image: _torch.Tensor, target: _torch.Tensor) -> Tuple[_torch.Tensor, _torch.Tensor]:
        return image, target


class SnowDatasetContextPrediction(SnowDatasetSelfSupervisedLearning):
    def __init__(self) -> None:
        super().__init__()

    def transform(self, image: _torch.Tensor, target: _torch.Tensor) -> Tuple[_torch.Tensor, _torch.Tensor]:
        return image, target


class SnowDatasetMaskRegionPrediction(SnowDatasetSelfSupervisedLearning):
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
