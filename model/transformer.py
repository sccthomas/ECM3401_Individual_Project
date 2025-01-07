import abc as _abc
import typing as _t

import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F

import model.attention as _attention


# Note: The below class is only intended to be used through inheritance.
class _TransformerBlock(_abc.ABC, _nn.Module):
    """
    Transformer block module for the encoder stage in the HRViT-Swin-Segmentation model.

    """

    def __init__(
            self,
            *,
            in_patches: int,
            in_channels: int,
            patch_resolution: _t.Tuple[int, int],
            iterations: int,
            num_attention_heads: int,
            window_size: _t.Tuple[int, int],
            shifted_window: bool,
            dropout: bool,
    ) -> None:
        """

        :param in_patches: Number of patches.
        :param in_channels: Length of the patch embeddings.
        :param patch_resolution: Resolution of the input image after patch embeddings.
        :param iterations: Number of transformer block iterations.
        :param num_attention_heads: Number of attention heads.
        :param window_size: Size of the attention window.
        :param shifted_window: Use shifted window.
        :param dropout: Use dropout.
        """
        super(_TransformerBlock, self).__init__()
        self.__in_patches, self.__in_channels = in_patches, in_channels

        hidden_channels = in_channels * 2
        self._iterations = _nn.ModuleList(
            [
                _nn.ModuleDict(
                    {
                        'attention': _attention.SwinTransformerAttention(
                            in_patches=in_patches,
                            in_channels=in_channels,
                            dropout=dropout,
                            num_attention_heads=num_attention_heads,
                            patch_resolution=patch_resolution,
                            shifted_window=shifted_window if i % 2 == 0 and i != 0 else False,
                            window_size=window_size,
                        ),
                        'norm1': _nn.LayerNorm(in_channels),
                        'norm2': _nn.LayerNorm(in_channels),
                        'mlp': _nn.Sequential(
                            _nn.Linear(in_channels, hidden_channels),
                            _nn.Dropout(0.1) if dropout else _nn.Identity(),
                            _nn.GELU(),
                            _nn.Linear(hidden_channels, in_channels),
                            _nn.Dropout(0.1) if dropout else _nn.Identity(),
                        )
                    }
                )
                for i in range(iterations)
            ]
        )

    @property
    def in_patches(self) -> int:
        """

        :return: Number of patches.
        """
        return self.__in_patches

    @property
    def in_channels(self) -> int:
        """

        :return: Input channels.
        """
        return self.__in_channels

    def forward(self, patch_embeddings: _torch.Tensor) -> _torch.Tensor:
        """
        Forward pass of the transformer block module during encoder stage.

        :param patch_embeddings: Patch embeddings to pass through the transformer block.
        :return: Transformed patch embeddings.
        """
        in_patches = self.in_patches
        in_channels = self.in_channels
        assert patch_embeddings.shape[1:] == (in_patches, in_channels), \
            (f"Expected shape {(in_patches, in_channels)}, got {tuple(patch_embeddings.shape[1:])} before "
             f"`TransformerBlock`.")

        iterations = self._iterations

        # Pass the patch embeddings through the transformer block iterations.
        for iteration in iterations:
            attention_output = iteration['attention'](iteration['norm1'](patch_embeddings))
            patch_embeddings = patch_embeddings + attention_output  # Add out-of-place

            mlp_output = iteration['mlp'](iteration['norm2'](patch_embeddings))
            patch_embeddings = patch_embeddings + mlp_output  # Add out-of-place

        patch_embeddings = self._post_process(patch_embeddings)

        return patch_embeddings

    @_abc.abstractmethod
    def _post_process(self, patch_embeddings: _torch.Tensor) -> _torch.Tensor:
        """
        Post-processing of the patch embeddings.

        :param patch_embeddings: Patch embeddings.
        :return: Post processed patch embeddings.
        """


class TransformerBlockEncoder(_TransformerBlock):
    """
    Transformer block module for the encoder stage in the HRViT-Swin-Segmentation model.
    """

    def __init__(
            self,
            *,
            in_patches: int,
            in_channels: int,
            patch_resolution: _t.Tuple[int, int],
            iterations: int,
            num_attention_heads: int,
            window_size: _t.Tuple[int, int],
            shifted_window: bool,
            dropout: bool,
    ) -> None:
        """

        :param in_patches: Number of patches.
        :param in_channels: Length of the patch embeddings.
        :param patch_resolution: Resolution of the input image after patch embeddings.
        :param iterations: Number of transformer block iterations.
        :param num_attention_heads: Number of attention heads.
        :param window_size: Size of the attention window.
        :param shifted_window: Use shifted window.
        :param dropout: Use dropout.
        """
        super(TransformerBlockEncoder, self).__init__(
            in_patches=in_patches,
            in_channels=in_channels,
            patch_resolution=patch_resolution,
            iterations=iterations,
            num_attention_heads=num_attention_heads,
            window_size=window_size,
            shifted_window=shifted_window,
            dropout=dropout,
        )

    def _post_process(self, patch_embeddings: _torch.Tensor) -> _torch.Tensor:
        """
        Post-processing of the patch embeddings to asser the shape.

        :param patch_embeddings: Patch embeddings.
        :return: Post processed patch embeddings.
        """
        dimensions = (self.in_patches, self.in_channels)

        assert patch_embeddings.shape[1:] == dimensions, \
            f"Expected shape {dimensions}, got {patch_embeddings.shape[1:]} after `TransformerBlockEncoder`."

        return patch_embeddings


class TransformerBlockDecoder(_TransformerBlock):
    """
    Transformer block module for the decoder stage in the HRViT-Swin-Segmentation model.
    """

    def __init__(
            self,
            *,
            in_patches: int,
            in_channels: int,
            patch_resolution: _t.Tuple[int, int],
            output_dims: _t.Tuple[int, int],
            iterations: int,
            num_attention_heads: int,
            window_size: _t.Tuple[int, int],
            shifted_window: bool,
            dropout: bool,
    ) -> None:
        """

        :param in_patches: Number of patches.
        :param in_channels: Length of the patch embeddings.
        :param patch_resolution: Resolution of the input image after patch embeddings.
        :param output_dims: Output dimensions.
        :param iterations: Number of transformer block iterations.
        :param num_attention_heads: Number of attention heads.
        :param window_size: Size of the attention window.
        :param shifted_window: Use shifted window.
        :param dropout: Use dropout.
        """
        super(TransformerBlockDecoder, self).__init__(
            in_patches=in_patches,
            in_channels=in_channels,
            patch_resolution=patch_resolution,
            iterations=iterations,
            num_attention_heads=num_attention_heads,
            window_size=window_size,
            shifted_window=shifted_window,
            dropout=dropout,
        )

        output_num_patches, output_vector_len = output_dims
        self.__linear_operation = _nn.Linear(in_channels, output_vector_len)
        self.__output_num_patches, self.__output_vector_len = output_num_patches, output_vector_len

    def _post_process(self, patch_embeddings: _torch.Tensor) -> _torch.Tensor:
        """
        Post-processing of the patch embeddings to upsample the patch embeddings to the output dimensions and assert the
        shape.

        :param patch_embeddings: Patch embeddings.
        :return: Post processed patch embeddings.
        """
        linear_operation = self.__linear_operation
        output_num_patches, output_vector_len = self.__output_num_patches, self.__output_vector_len

        # Linear operation to project the patch embeddings to the output dimensions.
        if output_num_patches != patch_embeddings.shape[1] and output_vector_len != patch_embeddings.shape[2]:
            patch_embeddings = linear_operation(patch_embeddings)
            patch_embeddings = _F.interpolate(
                input=patch_embeddings.permute(0, 2, 1),
                size=(output_num_patches,),
                mode='nearest',
            ).permute(0, 2, 1).contiguous()

        assert patch_embeddings.shape[1:] == (output_num_patches, output_vector_len), \
            (f"Expected shape {(output_num_patches, output_vector_len)}, got {patch_embeddings.shape[1:]} after "
             f"`TransformerBlockDecoder`.")

        return patch_embeddings
