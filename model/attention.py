import torch as _torch
import torch.nn as _nn
import typing as _t


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Swin Transformer Attention - Source https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
# Note - This is a modified version that is compatible for my use case.
# License - MIT License
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class SwinTransformerAttention(_nn.Module):
    def __init__(
            self,
            dropout: bool,
            num_heads: int,
            num_patches: int,
            shifted_window: bool,
            vector_len: int,
            window_size: _t.Tuple[int, int],
    ) -> None:
        super().__init__()
        self.__vector_len = vector_len
        self.__num_patches = num_patches
        self.__window_attention = _WindowAttention(vector_len, window_size, num_heads, dropout)
        self.__window_size = window_size

    def forward(self, patch_embeddings: _torch.Tensor) -> _torch.Tensor:
        vector_len = self.__vector_len
        num_patches = self.__num_patches
        window_attention = self.__window_attention
        window_size = self.__window_size

        # Partition the patch embeddings into non-overlapping windows
        patch_embeddings = _window_partition(patch_embeddings, window_size)

        # Apply the window attention mechanism
        patch_embeddings = window_attention(patch_embeddings)

        # Reconstruct the patch embeddings
        patch_embeddings = _window_reverse(patch_embeddings, window_size, num_patches, vector_len)

        return patch_embeddings


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Private Helpers
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class _WindowAttention(_nn.Module):
    """
    The window attention module that applies self-attention to non-overlapping windows of patch embeddings.
    """

    def __init__(self, vector_len: int, window_size: _t.Tuple[int, int], num_heads: int, dropout: bool) -> None:
        """

        :param vector_len: Patch embedding length
        :param window_size: Window size (height, width), patches are contained within each window
        :param num_heads: Number of attention heads
        :param dropout: Whether to apply dropout to the attention weights
        """
        super().__init__()
        head_dim = vector_len // num_heads
        window_size_h, window_size_w = window_size
        dropout_rate = 0.0 if dropout is False else 0.1

        self.__window_size = window_size  # Wh, Ww
        self.__num_heads = num_heads
        self.__scale = head_dim ** -0.5
        self.__qkv = _nn.Linear(vector_len, vector_len * 3, bias=True)
        self.__attn_drop = _nn.Dropout(dropout_rate)
        self.__proj = _nn.Linear(vector_len, vector_len)
        self.__proj_drop = _nn.Dropout(dropout_rate)
        self.__softmax = _nn.Softmax(dim=-1)

        # Define a parameter table of relative position bias
        relative_position_bias_table = _nn.Parameter(
            _torch.zeros((2 * window_size_h - 1) * (2 * window_size_w - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        _nn.init.trunc_normal_(relative_position_bias_table, std=.02)
        self.__relative_position_bias_table = relative_position_bias_table

        # Compute pairwise relative position index
        coords_h = _torch.arange(window_size_h)
        coords_w = _torch.arange(window_size_w)
        coords = _torch.stack(_torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = _torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size_h - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size_w - 1
        relative_coords[:, :, 0] *= 2 * window_size_w - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, windowed_patch_embeddings: _torch.Tensor, mask: _torch.Tensor = None):
        """

        :param windowed_patch_embeddings: The windowed patch embeddings, shape (num_windows*B, window_size, embed_len)
        :param mask: The attention mask used for shifted window attention, shape (num_windows*B, window_size, window_size)
        :return: The attended windowed patch embeddings, shape (num_windows*B, window_size, embed_len)
        """
        b, n, c = windowed_patch_embeddings.shape  # num_windows*B, window_size, embed_len
        q, k, v = (
            self.__qkv(windowed_patch_embeddings)
            .reshape(b, n, 3, self.__num_heads, c // self.__num_heads)
            .permute(2, 0, 3, 1, 4)
        )  # 3, num_windows*B, num_heads, window_size, head_dim

        q = q * self.__scale
        attn = (q @ k.transpose(-2, -1))  # num_windows*B, num_heads, window_size, window_size

        relative_position_bias = self.__relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.__window_size[0] * self.__window_size[1],
            self.__window_size[0] * self.__window_size[1],
            -1
        )  # Wh*Ww, Wh*Ww, num_heads
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # num_heads, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # num_windows*B, num_heads, window_size, window_size

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(b // nW, nW, self.__num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.__num_heads, n, n)
        attn = self.__softmax(attn)

        attn = self.__attn_drop(attn)

        windowed_patch_embeddings = (attn @ v).transpose(1, 2).reshape(b, n, c)  # num_windows*B, window_size, embed_len
        windowed_patch_embeddings = self.__proj(windowed_patch_embeddings)
        windowed_patch_embeddings = self.__proj_drop(windowed_patch_embeddings)

        return windowed_patch_embeddings


def _window_partition(x: _torch.Tensor, window_size: _t.Tuple[int, int]) -> _torch.Tensor:
    """
    Partition patch embeddings into non-overlapping windows along the patch sequence.

    :param x: The patch embeddings, shape (B, num_patches, embed_len)
    :param window_size: The size of the window
    :return: The partitioned windows, shape (num_windows*B, window_size, embed_len)
    """
    B, num_patches, embed_len = x.shape
    window_size_h, window_size_w = window_size
    amount_windows = window_size_h * window_size_w

    # Calculate the number of windows per batch
    num_windows = num_patches // amount_windows  # This assumes num_patches is divisible by window_size

    # Reshape the tensor to (B, num_windows, window_size, embed_len)
    x = x.view(B, num_windows, amount_windows, embed_len)  # Shape: (B, num_windows, window_size, embed_len)

    # Reshape to get the final output shape (num_windows * B, window_size, embed_len)
    windows = x.view(-1, amount_windows, embed_len)

    return windows


def _window_reverse(
        x: _torch.Tensor,
        window_size: _t.Tuple[int, int],
        patch_len: int,
        embed_len: int,
) -> _torch.Tensor:
    """
    Reverse the window partition operation and reconstruct the original patch embeddings.

    :param x: The partitioned windows, shape (num_windows*B, window_size, embed_len)
    :param window_size: The size of the window (i.e., number of patches per window)
    :param patch_len: The total number of patches per batch (B, num_patches, embed_len)
    :param embed_len: The embedding length of each patch
    :return: The reconstructed patch embeddings, shape (B, num_patches, embed_len)
    """
    window_size_h, window_size_w = window_size
    amount_windows = window_size_h * window_size_w

    B = x.shape[0] // (patch_len // amount_windows)  # Recalculate batch size from the number of windows and patches

    # Reshape the windows to (B, num_windows, window_size, embed_len)
    x = x.view(B, -1, amount_windows, embed_len)

    # Reassemble the windows back into patches (B, num_patches, embed_len)
    x_reconstructed = x.view(B, patch_len, embed_len)  # Reshape to (B, patch_len, embed_len)

    return x_reconstructed
