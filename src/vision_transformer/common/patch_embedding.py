import torch as _torch
import torch.nn as _nn


class PatchEmbedding(_nn.Module):
    """
    Patch embedding layer that will convert an image into patch embeddings.
    """

    def __init__(self, *, in_channels: int, embed_dim: int, patch_size: int, image_size: int) -> None:
        """

        :param in_channels: The number of input channels, i.e. RGB channels for image.
        :param embed_dim: The length to project patches into.
        :param patch_size: The size of each patch in the image.
        :param image_size: The size of the image.
        """
        super(PatchEmbedding, self).__init__()
        H = W = image_size // patch_size
        num_patches = H * W

        self.__projection = _nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.__positional_encoding = _nn.Parameter(_torch.zeros(1, num_patches, embed_dim))

        self.__H = H
        self.__W = W
        self.__num_patches = num_patches

        self.__initialize_weights()

    @property
    def num_patches(self) -> int:
        """
        Get the number of patches.

        :return: The number of patches.
        """
        return self.__num_patches

    @property
    def H(self) -> int:
        """
        Get the height of the patched image.

        :return: The height of the patched image.
        """
        return self.__H

    @property
    def W(self) -> int:
        """
        Get the width of the patched image.

        :return: The width of the patched image.
        """
        return self.__W

    @property
    def resolution(self) -> tuple[int, int]:
        """
        Get the resolution of the patched image.

        :return: The resolution of the patched image.
        """
        return self.__H, self.__W

    def forward(self, image: _torch.Tensor) -> _torch.Tensor:
        """
        Convert an image into patch embeddings.

        :param image: The input image tensor.
        :return: The patch embeddings.
        """
        projection = self.__projection
        positional_encoding = self.__positional_encoding

        patch_embeddings = projection(image)  # [B, embed_dim, H/patch_size, W/patch_size]
        patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2).contiguous()  # [B, H*W/P^2, embed_dim]
        patch_embeddings = patch_embeddings + positional_encoding

        return patch_embeddings

    def __initialize_weights(self) -> None:
        """
        Initialize the weights of the patch embedding layer.
        """
        projection = self.__projection

        _nn.init.xavier_uniform_(projection.weight)
        _nn.init.constant_(projection.bias, 0)
