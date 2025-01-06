import model.config as _config
import model.transformer as _transformer
import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F
import typing as _t


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

        :param num_stages:
        :param transformer_blocks:
        :param skip_connections:
        """
        super(Encoder, self).__init__()

        self.__num_stages = num_stages
        self.__skip_connections = skip_connections
        self.__transformer_blocks = transformer_blocks

    # Note - This is intended as the main entry to the class.
    @classmethod
    def from_config(cls, config: _config.EncoderConfig) -> 'Encoder':
        """
        Create encoder class from config.

        :param config: Configuration object containing all semantic segmentation model hyperparameters.
        """
        num_stages = config.num_stages
        patch_embedding_configs = config.patch_embedding_configs
        transformer_block_configs = config.transformer_block_configs

        # Create Skip Connection dimension information for each Patch Embedding.
        skip_connection_in_dimensions = tuple([
            tuple([
                (patch_embedding_config.in_patches, patch_embedding_config.in_channels)
                for patch_embedding_config in patch_embedding_configs[:i] + patch_embedding_configs[i + 1:]
            ])
            for i, _ in enumerate(patch_embedding_configs)
        ])

        # Create Transformer Block and Skip Connection for each Patch Embedding in the Stage.
        transformer_blocks = _nn.ModuleList()
        skip_connections = _nn.ModuleList()
        for stage in range(num_stages):
            transformer_blocks_stage = _nn.ModuleList()
            skip_connections_stage = _nn.ModuleList()
            # - Create Transformer Block and Skip Connection for each Patch Embedding in the Stage.
            for i, patch_embedding_config in enumerate(patch_embedding_configs):
                # - Patch Embedding information
                in_patches = patch_embedding_config.in_patches
                in_channels = patch_embedding_config.in_channels
                patch_resolution = patch_embedding_config.patch_resolution

                # - Create Transformer Block for Patch Embedding Stage.
                transformer_block_config = transformer_block_configs[i][stage]
                transformer_block = _transformer.TransformerBlockEncoder(
                    # - Patch Embedding information
                    in_patches=in_patches,
                    in_channels=in_channels,
                    patch_resolution=patch_resolution,
                    # - Transformer Block information
                    window_size=transformer_block_config.window_size,
                    num_attention_heads=transformer_block_config.num_attention_heads,
                    shifted_window=transformer_block_config.shifted_window,
                    iterations=transformer_block_config.iterations,
                    dropout=transformer_block_config.dropout,
                )
                transformer_blocks_stage.append(transformer_block)

                # - Create Skip connection for Transformer Block in Patch Embedding Stage.
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

        # - Manually feed the patch embeddings to the first stage of the encoder
        patch_embeddings = [
            transformer_block(patch_embedding)
            for transformer_block, patch_embedding in zip(transformer_blocks[0], patch_embeddings)
        ]

        # - Then feed the output of the first stage to the second stage, but fuse together patch embedding outputs from
        #   different scales.
        for i in range(1, num_stages):
            # - Fuse together patch embedding outputs from different scales
            patch_embeddings = [
                skip_connection(patch_embedding, patch_embeddings[:j] + patch_embeddings[j + 1:])
                for j, (skip_connection, patch_embedding) in enumerate(zip(skip_connections[i], patch_embeddings))
            ]
            # - Feed the fused patch embeddings to the next stage of the encoder
            patch_embeddings = [
                transformer_block(patch_embedding)
                for transformer_block, patch_embedding in zip(transformer_blocks[i], patch_embeddings)
            ]

        assert len_patch_embeddings == len(
            patch_embeddings), ("Number of patch embeddings should be the same after passing through all stages of the "
                                "encoder.")

        return patch_embeddings


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Protected Helpers
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


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

        kwargs = {'size': (out_patches,), 'mode': 'nearest'}
        for patch_embedding, linear_operation, in_dimension in zip(
                secondary_patch_embeddings,
                linear_operations,
                in_dimensions,
        ):
            # - Assert that the patch embedding aligns with the linear operation
            assert patch_embedding.shape[1:] == in_dimension, (
                "Patch embedding dimension does not align with the linear operation."
            )
            translated_patch_embedding = _F.interpolate(
                linear_operation(patch_embedding).permute(0, 2, 1),
                **kwargs
            ).permute(0, 2, 1)
            # - Assert that the translated patch embedding aligns with the target patch embedding
            assert translated_patch_embedding.shape[1:] == expected_shape, (
                "Translated patch embedding dimension does not align with the target patch embedding."
            )
            primary_patch_embedding += translated_patch_embedding

        # - Normalize the fused patch embeddings to reduce over active neurons.
        primary_patch_embedding = norm(primary_patch_embedding)

        return primary_patch_embedding
