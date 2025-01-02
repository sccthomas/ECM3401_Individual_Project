import torch as _torch
import torch.nn as _nn


class PatchEmbedding(_nn.Module):
    """
    Patch Embedding layer.
    - Converts image into patch embeddings.
    """

    def __init__(self, *, in_channels: int, embed_dim: int, patch_size: int, image_size: int) -> None:
        """

        :param in_channels: The number of input channels, i.e. RGB channels for image.
        :param embed_dim: The length to project patches into.
        :param patch_size: The size of each patch in the image.
        """
        super(PatchEmbedding, self).__init__()
        num_patches = (image_size // patch_size) ** 2

        self.__projection = _nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.__positional_encoding = _nn.Parameter(_torch.randn(1, num_patches, embed_dim))

    def forward(self, image: _torch.Tensor) -> _torch.Tensor:
        """
        Convert an image into patch embeddings.

        :param image: The input image tensor.
        :return: The patch embeddings.
        """
        projection = self.__projection
        positional_encoding = self.__positional_encoding

        patch_embeddings = projection(image)  # [B, embed_dim, H/patch_size, W/patch_size]
        patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)  # [B, H*W/P^2, embed_dim]
        patch_embeddings += positional_encoding

        return patch_embeddings
