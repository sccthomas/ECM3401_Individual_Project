import itertools as _iter
import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F
import typing as _t
import model.config as _config


class Encoder(_nn.Module):
    """
    Encoder module of the HRViT-Swin-Segmentation model.
    """

    def __init__(self, config: _config.ModelConfig) -> None:
        """

        :param config: Configuration object containing all semantic segmentation model hyperparameters.
        """
        super(Encoder, self).__init__()

        patch_embedding_dims = config.patch_embedding_dims
        num_stages = config.encoder.num_stages
        iterations = config.encoder.iterations

        self.__transformer_blocks = _nn.ModuleList([
            _nn.ModuleList([
                _TransformerBlock(vector_len=patch_embedding_dim.vector_len, iterations=iterations)
                for patch_embedding_dim in patch_embedding_dims
            ])
            for _ in range(num_stages)
        ])

        self.__skip_connections = _nn.ModuleList([
            _nn.ModuleList([
                _SkipConnections(
                    target_patch_embedding_dim=patch_embedding_dim,
                    patch_embedding_dims=_iter.chain(patch_embedding_dims[:i], patch_embedding_dims[i + 1:])
                )
                for i, patch_embedding_dim in enumerate(patch_embedding_dims)
            ])
            for _ in range(num_stages)
        ])

        self.__num_stages = num_stages

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
            target_patch_embedding_dim: '_config._Patch_Embedding_Dim',
            patch_embedding_dims: _t.Iterable['_config._Patch_Embedding_Dim']
    ) -> None:
        """

        :param target_patch_embedding_dim: The target patch embedding dimension to which all patch embeddings will be
               transformed to.
        :param patch_embedding_dims: List of patch embedding dimensions to translate to the target patch embedding
               dimension.
        """
        super(_SkipConnections, self).__init__()

        # Define Linear operations to convert patch embeddings to target patch embedding dimension
        self.__linear_operations = _nn.ModuleList([
            _nn.Linear(patch_embedding_dim.vector_len, target_patch_embedding_dim.vector_len)
            for patch_embedding_dim in patch_embedding_dims
        ])
        self.__patch_len = target_patch_embedding_dim.patch_len
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
        patch_len = self.__patch_len
        norm = self.__norm

        kwargs = {'size': (patch_len,), 'mode': 'nearest'}
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

    def __init__(self, vector_len: int, iterations: int) -> None:
        """

        :param vector_len: Patch embedding vector length.
        :param iterations: Number of transformer block iterations.
        """
        super(_TransformerBlock, self).__init__()
        hidden_dim = vector_len * 2

        self.__iterations = iterations
        self.__vector_len = vector_len
        self.__attention = lambda x: x  # Placeholder for attention mechanism IMPORTANT
        self.__norm1 = _nn.ModuleList([_nn.LayerNorm(vector_len) for _ in range(iterations)])
        self.__norm2 = _nn.ModuleList([_nn.LayerNorm(vector_len) for _ in range(iterations)])
        self.__mlp = _nn.ModuleList([
            _nn.Sequential(
                _nn.Linear(vector_len, hidden_dim),
                _nn.GELU(),
                _nn.Linear(hidden_dim, vector_len)
            )
            for _ in range(iterations)
        ])

    @property
    def vector_len(self) -> int:
        """
        :return: Patch embedding vector length.
        """
        return self.__vector_len

    @property
    def iterations(self) -> int:
        """
        :return: Number of transformer block iterations.
        """
        return self.__iterations

    def forward(self, patch_embeddings: _torch.Tensor) -> _torch.Tensor:
        """
        Forward pass of the transformer block module.

        :param patch_embeddings: Patch embeddings to pass through the transformer block.
        :return: Transformed patch embeddings.
        """
        attention = self.__attention
        iterations = self.__iterations
        norm1 = self.__norm1
        norm2 = self.__norm2
        mlp = self.__mlp

        for i in range(iterations):
            attn_output = attention(patch_embeddings)
            patch_embeddings = norm1[i](attn_output + patch_embeddings)
            ffn_output = mlp[i](patch_embeddings)

            patch_embeddings = norm2[i](ffn_output + patch_embeddings)

        return patch_embeddings
