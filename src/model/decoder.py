import math as _math
import typing as _t

import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F

import src.model.config as _config
import src.model.transformer as _transformer


class Decoder(_nn.Module):
    """
    Decoder module of the HRViT-Swin-Segmentation model.
    """

    def __init__(
            self,
            *,
            max_in_channels: int,
            num_classes: int,
            output_dimensions: _t.Tuple[int, int, int],
            transformer_blocks: '_nn.ModuleList[_transformer.TransformerBlockDecoder]',
    ) -> None:
        """

        :param max_in_channels: Max number of input channels to the decoder.
        :param num_classes: Number of classes to predict.
        :param output_dimensions: Tuple of (batch_size, output_height, output_width).
        :param transformer_blocks: List of transformer blocks in the decoder.
        """
        super(Decoder, self).__init__()
        self._batch_size, self.__output_height, self.__output_width = output_dimensions
        self.__transformer_blocks = transformer_blocks
        self.__prediction_head = _nn.Conv2d(
            in_channels=max_in_channels,
            out_channels=num_classes,
            kernel_size=1,
        )

    @classmethod
    def from_config(cls, config: _config.DecoderConfig) -> 'Decoder':
        """

        :param config:
        :return:
        """
        output_dimensions = config.output_dimensions
        max_in_channels = config.max_in_channels
        num_classes = config.num_classes

        patch_embedding_configs = config.patch_embedding_configs
        transformer_block_configs = config.transformer_block_configs
        transformer_blocks = _nn.ModuleList([
            _transformer.TransformerBlockDecoder(
                in_patches=patch_embedding_config.in_patches,
                in_channels=patch_embedding_config.in_channels,
                patch_resolution=patch_embedding_config.patch_resolution,
                output_dims=(next_patch_embedding_config.in_patches, next_patch_embedding_config.in_channels),
                iterations=transformer_block_config.iterations,
                num_attention_heads=transformer_block_config.num_attention_heads,
                window_size=transformer_block_config.window_size,
                shifted_window=transformer_block_config.shifted_window,
                dropout=transformer_block_config.dropout,
            )
            for i, (patch_embedding_config, transformer_block_config) in enumerate(
                zip(patch_embedding_configs[:-1], transformer_block_configs[:-1])
            )
            if (next_patch_embedding_config := patch_embedding_configs[i + 1])
        ])
        final_transformer_block_config = transformer_block_configs[-1]
        final_patch_embedding_config = patch_embedding_configs[-1]
        in_patches = final_patch_embedding_config.in_patches
        in_channels = final_patch_embedding_config.in_channels
        transformer_blocks.append(
            _transformer.TransformerBlockDecoder(
                in_patches=in_patches,
                in_channels=in_channels,
                patch_resolution=final_patch_embedding_config.patch_resolution,
                output_dims=(in_patches, in_channels),
                iterations=final_transformer_block_config.iterations,
                num_attention_heads=final_transformer_block_config.num_attention_heads,
                window_size=final_transformer_block_config.window_size,
                shifted_window=final_transformer_block_config.shifted_window,
                dropout=final_transformer_block_config.dropout
            )
        )

        return cls(
            max_in_channels=max_in_channels,
            num_classes=num_classes,
            output_dimensions=output_dimensions,
            transformer_blocks=transformer_blocks,
        )

    def forward(self, patch_embeddings: _t.List[_torch.Tensor]) -> _torch.Tensor:
        """
        Forward pass of the decoder.

        :param patch_embeddings: List of patch embeddings to decode and upsample into the output dimensions.
        :return: A tensor of shape (batch_size, number_classes, output_dims_h, output_dims_w).
        """
        output_height, output_width = self.__output_height, self.__output_width
        transformer_blocks = self.__transformer_blocks
        prediction_head = self.__prediction_head

        patch_embeddings_decoder = list(reversed(patch_embeddings))
        output = _torch.zeros_like(patch_embeddings_decoder[0])
        for patch_embedding, transformer_block in zip(patch_embeddings_decoder, transformer_blocks):
            assert output.shape[1:] == patch_embedding.shape[1:]
            output = transformer_block(output + patch_embedding)

        batch_size, num_patches, vector_len = output.shape

        grid_size = int(_math.sqrt(num_patches))
        assert grid_size ** 2 == num_patches

        output = output.reshape(batch_size, vector_len, grid_size, grid_size).contiguous()
        output = _F.interpolate(
            input=output,
            size=(output_height, output_width),
            mode='bilinear',
            align_corners=False,
        )

        output = prediction_head(output)

        assert output.shape[1:] == (1, output_height, output_width)

        return output
