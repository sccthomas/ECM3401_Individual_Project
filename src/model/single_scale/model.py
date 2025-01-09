import torch as _torch
import torch.nn as _nn

import src.model.single_scale.encoder as _encoder
import src.model.single_scale.patch_embedding as _patch_embedding


class SemanticSegmentationVisionTransformer(_nn.Module):
    """
    Semantic Segmentation Vision Transformer model.
    """

    def __init__(self, in_channels, num_classes, embed_dim, patch_size, img_size, num_heads, num_layers) -> None:
        """

        :param in_channels: Number of input channels.
        :param num_classes: Number of classes.
        :param embed_dim: Embedding dimension.
        :param patch_size: Patch size.
        :param img_size: Image size.
        :param num_heads: Number of heads.
        :param num_layers: Number of layers.
        """
        super(SemanticSegmentationVisionTransformer, self).__init__()
        self.__patch_embedding = _patch_embedding.PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            image_size=img_size,
        )
        self.__encoder = _encoder.TransformerEncoder(embed_dim, num_heads, num_layers=num_layers, mlp_ratio=4.0,
                                                     dropout=0.1)
        self.__decoder = _nn.Sequential(
            _nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            _nn.ReLU(),
            _nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # Upsample 16x16 -> 32x32
            _nn.ReLU(),
            _nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # Upsample 32x32 -> 64x64
            _nn.ReLU(),
            _nn.ConvTranspose2d(64, num_classes, kernel_size=4, stride=4),  # Upsample 64x64 -> 256x256
        )
        self.__img_size = img_size
        self.__patch_size = patch_size

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        """
        Forward pass.

        :param x: Input tensor.
        :return: Output tensor.
        """
        # Patch embedding
        patches = self.__patch_embedding(x)  # (B, num_patches, embed_dim)

        # Transformer Encoder
        encoded = self.__encoder(patches)  # (B, num_patches, embed_dim)

        # Reshape and decode
        H, W = self.__img_size // self.__patch_size, self.__img_size // self.__patch_size
        encoded = encoded.transpose(1, 2).view(-1, encoded.shape[-1], H, W)  # (B, embed_dim, H, W)
        output = self.__decoder(encoded)  # (B, num_classes, img_size, img_size)

        return output
