import typing as _t

import torch as _torch
import torch.nn as _nn


class Decoder(_nn.Module):
    """
    Decoder module that will upsample the final embeddings to the output dimensions.
    """

    def __init__(self, resolution: int, transposed_convolutions: _nn.Sequential) -> None:
        """

        :param resolution: Resolution of the image after applying patch embeddings.
        :param transposed_convolutions: Transposed convolutions to upsample the embeddings.
        """
        super(Decoder, self).__init__()
        self.__resolution = resolution
        self.__transposed_convolutions = transposed_convolutions

    @classmethod
    def create(cls, final_num_patches: int, final_embed_dim: int, output_dims: _t.Tuple[int, int, int]) -> 'Decoder':
        """
        Create a decoder that will upsample the final embeddings to the output dimensions.

        :param final_num_patches:
        :param final_embed_dim:
        :param output_dims:
        :return:
        """
        # Compute the number of operations required to reach the final resolution
        resolution = int(final_num_patches ** 0.5)
        resolution_ = resolution
        num_classes, H, W = output_dims
        num_operations = 0
        while resolution_ < H:
            resolution_ *= 2
            num_operations += 1

        # Compute the best factor to reduce the final embedding dimension
        best_factor = final_embed_dim // num_operations
        while final_embed_dim % best_factor != 0:
            best_factor += 1

        # Compute the dimensions of the transposed convolutions
        dim = final_embed_dim
        transposed_dims = [dim]
        for i in range(num_operations):
            dim_ = dim - best_factor
            if dim_ <= 0:
                dim //= 2
            else:
                dim = dim_
            transposed_dims.append(dim)

        # Create the transposed convolutions
        transposed_convolutions = _nn.Sequential()
        for i in range(len(transposed_dims) - 1):
            dim_1 = transposed_dims[i]
            dim_2 = transposed_dims[i + 1]
            transposed_convolutions.append(
                _nn.ConvTranspose2d(dim_1, dim_2, kernel_size=2, stride=2)
            )
            transposed_convolutions.append(_nn.ReLU())

        # Add the final transposed convolution to predict the number of classes and a ReLU activation
        transposed_convolutions.append(
            _nn.ConvTranspose2d(transposed_dims[-1], num_classes, kernel_size=1, stride=1)
        )
        transposed_convolutions.append(_nn.ReLU())

        return cls(resolution=resolution, transposed_convolutions=transposed_convolutions)

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        """
        Forward pass of the decoder.

        :param x: Patch embedding to upsample to the output dimensions.
        :return: Predicted output tensor.
        """
        resolution = self.__resolution
        transposed_convolutions = self.__transposed_convolutions

        x = x.transpose(1, 2).reshape(-1, x.shape[-1], resolution, resolution)
        x = transposed_convolutions(x)

        return x
