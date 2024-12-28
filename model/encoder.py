import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F
import typing as _t
import model.config as _config
import model.attention as _attention


class Encoder(_nn.Module):
    """
    Encoder module of the HRViT-Swin-Segmentation model.
    """

    def __init__(
            self,
            *,
            num_stages: int,
            transformer_blocks: '_nn.ModuleList[_nn.ModuleList[_TransformerBlock]]',
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

        transformer_blocks = _nn.ModuleList()
        skip_connections = _nn.ModuleList()

        for stage in range(num_stages):
            transformer_blocks_stage = _nn.ModuleList()
            skip_connections_stage = _nn.ModuleList()
            for i, patch_embedding_config in enumerate(patch_embedding_configs):
                transformer_block_config = patch_embedding_config.transformer_block_configs[stage]
                transformer_block = _TransformerBlock(
                    vector_len=patch_embedding_config.vector_len,
                    num_patches=patch_embedding_config.num_patches,
                    iterations=transformer_block_config.iterations,
                    window_size=transformer_block_config.window_size,
                    num_heads=transformer_block_config.num_heads,
                    dropout=transformer_block_config.dropout,
                    shifted_window=transformer_block_config.shifted_window,
                )
                transformer_blocks_stage.append(transformer_block)
                skip_connections_dims = {
                    "target_patch_embedding_dim": patch_embedding_config,
                    "patch_embedding_dims": patch_embedding_configs[:i] + patch_embedding_configs[i + 1:]
                }
                skip_connection = _SkipConnections(
                    **skip_connections_dims
                )
                skip_connections_stage.append(skip_connection)

            transformer_blocks.append(transformer_blocks_stage)
            skip_connections.append(skip_connections_stage)

        return cls(num_stages=num_stages, transformer_blocks=transformer_blocks, skip_connections=skip_connections)

    @property
    def transformer_blocks(self) -> '_nn.ModuleList[_nn.ModuleList[_TransformerBlock]]':
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

    def forward(self, patch_embeddings: _t.Iterable[_torch.Tensor]) -> _t.Iterable[_torch.Tensor]:
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

        return patch_embeddings


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Private Helpers
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class _SkipConnections(_nn.Module):
    """
    Skip connections module to fuse together patch embedding outputs from different scales.
    """

    def __init__(
            self,
            target_patch_embedding_dim: _config.PatchEmbeddingConfig,
            patch_embedding_dims: _t.Iterable[_config.PatchEmbeddingConfig]
    ) -> None:
        """

        :param target_patch_embedding_dim: The target patch embedding dimension to which all patch embeddings will be
               transformed to.
        :param patch_embedding_dims: Iterable of patch embedding dimensions to translate to the target patch embedding
               dimension.
        """
        super(_SkipConnections, self).__init__()

        # Define Linear operations to convert patch embeddings to target patch embedding dimension
        self.__linear_operations = _nn.ModuleList([
            _nn.Linear(patch_embedding_dim.vector_len, target_patch_embedding_dim.vector_len)
            for patch_embedding_dim in patch_embedding_dims
        ])
        self.__num_patches = target_patch_embedding_dim.num_patches
        self.__norm = _nn.LayerNorm(target_patch_embedding_dim.vector_len)

    @property
    def linear_operations(self) -> '_nn.ModuleList[_nn.Linear]':
        """
        :return: List of linear operations to convert patch embeddings to the target patch embedding dimension.
        """
        return self.__linear_operations

    def forward(self, target_patch_embedding: _torch.Tensor, patch_embeddings: _torch.Tensor) -> _torch.Tensor:
        """
        Forward pass of the skip connections module.
        - Convert all patch embeddings to the target patch embedding dimension.
        - Fuse together the converted patch embeddings with the target patch embedding.

        :param target_patch_embedding: The target patch embedding to fuse with the converted patch embeddings.
        :param patch_embeddings: A list of patch embeddings to convert to the target patch embedding dimension.
        :return: The fused patch embeddings.
        """
        linear_operations = self.__linear_operations
        num_patches = self.__num_patches
        norm = self.__norm

        kwargs = {'size': (num_patches,), 'mode': 'nearest'}
        for patch_embedding, linear_operation in zip(patch_embeddings, linear_operations):
            target_patch_embedding += _F.interpolate(
                linear_operation(patch_embedding).permute(0, 2, 1),
                **kwargs
            ).permute(0, 2, 1)

        target_patch_embedding = norm(target_patch_embedding)

        return target_patch_embedding


class _TransformerBlock(_nn.Module):
    """
    Transformer block module for the encoder stage in the HRViT-Swin-Segmentation model.

    """

    def __init__(
            self,
            dropout: bool,
            iterations: int,
            num_heads: int,
            num_patches: int,
            shifted_window: bool,
            vector_len: int,
            window_size: _t.Tuple[int, int],
    ) -> None:
        """

        :param dropout:
        :param iterations:
        :param num_heads:
        :param num_patches:
        :param shifted_window:
        :param vector_len:
        :param window_size:
        """
        super(_TransformerBlock, self).__init__()
        hidden_dim = vector_len * 2
        self.__vector_len = vector_len

        self.__iterations = _nn.ModuleList(
            [
                _nn.ModuleDict(
                    {
                        'attention': _attention.SwinTransformerAttention(
                            dropout,
                            num_heads,
                            num_patches,
                            shifted_window,
                            vector_len,
                            window_size,
                        ),
                        'norm1': _nn.LayerNorm(vector_len),
                        'norm2': _nn.LayerNorm(vector_len),
                        'mlp': _nn.Sequential(
                            _nn.Linear(vector_len, hidden_dim),
                            _nn.GELU(),
                            _nn.Linear(hidden_dim, vector_len)
                        )
                    }
                )
                for _ in range(iterations)
            ]
        )

    @property
    def vector_len(self) -> int:
        return self.__vector_len

    def forward(self, patch_embeddings: _torch.Tensor) -> _torch.Tensor:
        """
        Forward pass of the transformer block module.

        :param patch_embeddings: Patch embeddings to pass through the transformer block.
        :return: Transformed patch embeddings.
        """
        iterations = self.__iterations

        for iteration in iterations:
            attn_output = iteration['attention'](patch_embeddings)
            patch_embeddings = iteration['norm1'](attn_output + patch_embeddings)
            ffn_output = iteration['mlp'](patch_embeddings)
            patch_embeddings = iteration['norm2'](ffn_output + patch_embeddings)

        return patch_embeddings
