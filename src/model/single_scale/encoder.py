import torch as _torch
import torch.nn as _nn


class TransformerEncoder(_nn.Module):
    """
    Transformer encoder module.
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float, num_layers) -> None:
        """

        :param embed_dim: Patch embedding dimension.
        :param num_heads: Number of attention heads.
        :param mlp_ratio: Multi-layer perceptron ratio.
        :param dropout: Dropout rate.
        :param num_layers: Number of transformer encoder layers.
        """
        super(TransformerEncoder, self).__init__()
        self.layers = _nn.ModuleList([
            _nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu'
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        """
        Forward pass of the transformer encoder

        :param x: input tensor of shape (B, num_patches, embed_dim)
        :return:  output tensor of shape (B, num_patches, embed_dim)
        """
        for layer in self.layers:
            x = layer(x)
        return x
