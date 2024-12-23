import torch as _torch
import torch.nn as _nn


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Private Helpers
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class WindowAttention(_nn.Module):
    def __init__(self, vector_len, window_size, num_heads, dropout=False) -> None:
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
            _torch.zeros((2 * window_size_h - 1) * (2 * window_size_w - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        _nn.init.trunc_normal_(relative_position_bias_table, std=.02)
        self.__relative_position_bias_table = relative_position_bias_table

        # Get pair-wise relative position index for each token inside the window
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

    def forward(self, windowed_patch_embeddings, mask=None):
        """
        Args:
            windowed_patch_embeddings: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b, n, c = windowed_patch_embeddings.shape
        q, k, v = (
            self.__qkv(windowed_patch_embeddings)
            .reshape(b, n, 3, self.__num_heads, c // self.__num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q = q * self.__scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.__relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.__window_size[0] * self.__window_size[1], self.__window_size[0] * self.__window_size[1],
            -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(b // nW, nW, self.__num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.__num_heads, n, n)
            attn = self.__softmax(attn)
        else:
            attn = self.__softmax(attn)

        attn = self.__attn_drop(attn)

        windowed_patch_embeddings = (attn @ v).transpose(1, 2).reshape(b, n, c)
        windowed_patch_embeddings = self.__proj(windowed_patch_embeddings)
        windowed_patch_embeddings = self.__proj_drop(windowed_patch_embeddings)

        return windowed_patch_embeddings


def _window_partition(x: _torch.Tensor, window_size: int) -> _torch.Tensor:
    """
    Partition patch embeddings into non-overlapping windows along the patch sequence.

    :param x: The patch embeddings, shape (B, num_patches, embed_len)
    :param window_size: The size of the window (i.e., number of patches per window)
    :return: The partitioned windows, shape (num_windows*B, window_size, embed_len)
    """
    B, num_patches, embed_len = x.shape

    # Calculate the number of windows per batch
    num_windows = num_patches // window_size  # This assumes num_patches is divisible by window_size

    # Reshape the tensor to (B, num_windows, window_size, embed_len)
    x = x.view(B, num_windows, window_size, embed_len)  # Shape: (B, num_windows, window_size, embed_len)

    # Reshape to get the final output shape (num_windows * B, window_size, embed_len)
    windows = x.view(-1, window_size, embed_len)

    return windows


def _window_reverse(windows: _torch.Tensor, window_size: int, num_patches: int, embed_len: int) -> _torch.Tensor:
    """
    Reverse the window partition operation and reconstruct the original patch embeddings.

    :param windows: The partitioned windows, shape (num_windows*B, window_size, embed_len)
    :param window_size: The size of the window (i.e., number of patches per window)
    :param num_patches: The total number of patches per batch (B, num_patches, embed_len)
    :param embed_len: The embedding length of each patch
    :return: The reconstructed patch embeddings, shape (B, num_patches, embed_len)
    """
    B = windows.shape[0] // (
            num_patches // window_size)  # Recalculate batch size from the number of windows and patches

    # Reshape the windows to (B, num_windows, window_size, embed_len)
    windows = windows.view(B, -1, window_size, embed_len)

    # Reassemble the windows back into patches (B, num_patches, embed_len)
    x_reconstructed = windows.view(B, num_patches, embed_len)  # Reshape to (B, num_patches, embed_len)

    return x_reconstructed
