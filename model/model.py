import typing as _t
import torch as _torch
import torch.nn as _nn
import model.encoder as _tb
import model.patch_embedding as _pe
import itertools as _iter


class HRViT_Swin_Segmentation(_nn.Module):
    def __init__(self, in_channels: int, patch_scales: _t.List[int], embed_dims: _t.List[int]) -> None:
        super(HRViT_Swin_Segmentation, self).__init__()

        # Patch Embeddings Layer
        kwargs = {'in_channels': in_channels, 'image_size': IMAGE_SIZE}
        self.__multiscale_patch_embeddings = [
            _pe.PatchEmbedding(embed_dim=embed_dim, patch_size=patch_scale, **kwargs)
            for patch_scale, embed_dim in zip(patch_scales, embed_dims)
        ]

        # Encoder Layers
        self.__encoder_transformer_blocks = [
            [
                _tb.TransformerBlock(embed_dim=embed_dim)
                for embed_dim in embed_dims
            ]
            for _ in range(NUM_STAGES)
        ]

        # Decoder Layers

    def forward(self, image: _torch.Tensor) -> _torch.Tensor:
        multiscale_patch_embeddings = self.__multiscale_patch_embeddings
        encoder_transformer_blocks = self.__encoder_transformer_blocks

        # Patch Embeddings
        patch_embeddings = [
            patch_embedding(image)
            for patch_embedding in multiscale_patch_embeddings
        ]

        # Encoder Layers
        # - Feed the patch embeddings to the first stage of the encoder
        patch_embeddings = [
            encoder_block(patch_embedding)
            for patch_embedding, encoder_block in zip(patch_embeddings, encoder_transformer_blocks[0])
        ]
        # - Then feed the output of the first stage to the second stage, but fuse together patch embedding outputs from
        #   different scales.
        patch_embeddings = [
            encoder_block()
            for i, (patch_embedding, encoder_block) in enumerate(zip(patch_embeddings, encoder_transformer_blocks[1]))
        ]


# --------------------------------------------
# Private Helpers
# --------------------------------------------


IMAGE_SIZE = 512
NUM_STAGES = 3
