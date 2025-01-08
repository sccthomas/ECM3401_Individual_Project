import typing as _t

import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F

import src.model.config as _config
import src.model.transformer as _transformer


class Encoder(_nn.Module):
    """
    Encoder module of the HRViT-Swin-Segmentation model.
    """

    def __init__(
            self,
            *,
            num_stages: int,
            transformer_blocks: '_nn.ModuleList[_nn.ModuleList[_transformer.TransformerBlockEncoder]]',
            skip_connections: '_nn.ModuleList[_nn.ModuleList[_SkipConnections]]'
    ) -> None:
        """
        :param num_stages: The number of stages in the encoder.
        :param transformer_blocks: List of transformer blocks in each stage of the encoder.
        :param skip_connections: List of skip connections in each stage of the encoder.
        """
        super(Encoder, self).__init__()

        self.__num_stages = num_stages
        self.__skip_connections = skip_connections
        self.__transformer_blocks = transformer_blocks

    @classmethod
    def from_config(cls, config: _config.EncoderConfig) -> 'Encoder':
        """
        Create encoder class from config.

        :param config: Configuration object containing all semantic segmentation model hyperparameters.
        """
        num_stages = config.num_stages
        patch_embedding_configs = config.patch_embedding_configs
        transformer_block_configs = config.transformer_block_configs

        skip_connection_in_dimensions = tuple([
            tuple([
                (patch_embedding_config.in_patches, patch_embedding_config.in_channels)
                for patch_embedding_config in patch_embedding_configs[:i] + patch_embedding_configs[i + 1:]
            ])
            for i, _ in enumerate(patch_embedding_configs)
        ])

        transformer_blocks = _nn.ModuleList()
        skip_connections = _nn.ModuleList()
        for stage in range(num_stages):
            transformer_blocks_stage = _nn.ModuleList()
            skip_connections_stage = _nn.ModuleList()
            for i, patch_embedding_config in enumerate(patch_embedding_configs):
                in_patches = patch_embedding_config.in_patches
                in_channels = patch_embedding_config.in_channels
                patch_resolution = patch_embedding_config.patch_resolution

                transformer_block_config = transformer_block_configs[i][stage]
                transformer_block = _transformer.TransformerBlockEncoder(
                    in_patches=in_patches,
                    in_channels=in_channels,
                    patch_resolution=patch_resolution,
                    window_size=transformer_block_config.window_size,
                    num_attention_heads=transformer_block_config.num_attention_heads,
                    shifted_window=transformer_block_config.shifted_window,
                    iterations=transformer_block_config.iterations,
                    dropout=transformer_block_config.dropout,
                )
                transformer_blocks_stage.append(transformer_block)

                skip_connection = _SkipConnections(
                    out_patches=in_patches,
                    out_channels=in_channels,
                    in_dimensions=skip_connection_in_dimensions[i],
                )
                skip_connections_stage.append(skip_connection)

            transformer_blocks.append(transformer_blocks_stage)
            skip_connections.append(skip_connections_stage)

        return cls(num_stages=num_stages, transformer_blocks=transformer_blocks, skip_connections=skip_connections)

    @property
    def transformer_blocks(self) -> '_nn.ModuleList[_nn.ModuleList[_transformer.TransformerBlockEncoder]]':
        """
        :return: List of transformer blocks in each stage of the encoder.
        """
        return self.__transformer_blocks

    @property
    def skip_connections(self) -> '_nn.ModuleList[_nn.ModuleList[_SkipConnections]]':
        """
        :return: List of skip connections in each stage of the encoder.
        """
        return self.__skip_connections

    def forward(self, patch_embeddings: _t.List[_torch.Tensor]) -> _t.List[_torch.Tensor]:
        """
        Forward pass of the encoder module.
        - Feed the patch embeddings to the first stage of the encoder
        - Then feed the output of the first stage to the second stage, but fuse together patch embedding outputs from
          different scales.
        - Repeat the above step for all stages of the encoder.

        :param patch_embeddings: List of patch embeddings from the patch embedding layer.
        :return: List of patch embeddings after passing through all stages of the encoder.
        """
        num_stages = self.__num_stages
        skip_connections = self.__skip_connections
        transformer_blocks = self.__transformer_blocks

        len_patch_embeddings = len(patch_embeddings)

        patch_embeddings_encoded = []
        for i in range(len_patch_embeddings):
            patch_embedding = patch_embeddings[i]
            transformer_block = transformer_blocks[0][i]
            patch_embeddings_encoded.append(transformer_block(patch_embedding))

        for i in range(1, num_stages):
            for j in range(len_patch_embeddings):
                patch_embedding = patch_embeddings_encoded[j]
                skip_connection = skip_connections[i][j]
                patch_embeddings_encoded[j] = skip_connection(
                    patch_embedding, patch_embeddings_encoded[:j] + patch_embeddings_encoded[j + 1:])

            for j in range(len_patch_embeddings):
                patch_embedding = patch_embeddings_encoded[j]
                transformer_block = transformer_blocks[i][j]
                patch_embeddings_encoded[j] = transformer_block(patch_embedding)

        assert len_patch_embeddings == len(patch_embeddings_encoded)

        return patch_embeddings_encoded


class _SkipConnections(_nn.Module):
    """
    Skip connections module to fuse together patch embedding outputs from different scales.
    """

    def __init__(
            self,
            out_patches: int,
            out_channels: int,
            in_dimensions: _t.Tuple[int, int],
    ) -> None:
        """
        :param out_patches: Number of patches to output.
        :param out_channels: Number of channels in each patch embedding output.
        :param in_dimensions: List of tuples containing the number of patches and channels in each secondary patch
                              embedding (in_patches, in_channels).
        """
        super(_SkipConnections, self).__init__()

        self.__linear_operations = _nn.ModuleList([
            _nn.Linear(in_features=in_channels, out_features=out_channels)
            for (_, in_channels) in in_dimensions
        ])
        self.__out_patches = out_patches
        self.__out_channels = out_channels
        self.__in_dimensions = in_dimensions
        self.__norm = _nn.LayerNorm(out_channels)

    @property
    def linear_operations(self) -> '_nn.ModuleList[_nn.Linear]':
        """
        :return: List of linear operations to convert patch embeddings to the target patch embedding dimension.
        """
        return self.__linear_operations

    def forward(
            self,
            primary_patch_embedding: _torch.Tensor,
            secondary_patch_embeddings: _torch.Tensor,
    ) -> _torch.Tensor:
        """
        Forward pass of the skip connections module.
        - Convert all patch embeddings to the target patch embedding dimension.
        - Fuse together the converted patch embeddings with the target patch embedding.

        :param primary_patch_embedding: Primary patch embedding to fuse with the secondary patch embeddings.
        :param secondary_patch_embeddings: List of secondary patch embeddings to fuse with the primary patch embedding.
        :return: The fused patch embeddings.
        """
        linear_operations = self.__linear_operations
        out_patches = self.__out_patches
        out_channels = self.__out_channels
        in_dimensions = self.__in_dimensions
        norm = self.__norm

        expected_shape = (out_patches, out_channels)

        primary_patch_embedding_skip_connection = primary_patch_embedding.clone()
        kwargs = {'size': (out_patches,), 'mode': 'nearest'}
        for patch_embedding, linear_operation, in_dimension in zip(
                secondary_patch_embeddings,
                linear_operations,
                in_dimensions,
        ):
            assert patch_embedding.shape[1:] == in_dimension, (
                "Patch embedding dimension does not align with the linear operation."
            )
            translated_patch_embedding = linear_operation(patch_embedding).permute(0, 2, 1).contiguous()
            translated_patch_embedding = _F.interpolate(
                translated_patch_embedding,
                **kwargs
            ).permute(0, 2, 1).contiguous()
            assert translated_patch_embedding.shape[1:] == expected_shape, (
                "Translated patch embedding dimension does not align with the target patch embedding."
            )
            primary_patch_embedding_skip_connection = (
                    primary_patch_embedding_skip_connection + translated_patch_embedding
            )

        primary_patch_embedding_skip_connection = norm(primary_patch_embedding_skip_connection)

        return primary_patch_embedding_skip_connection
