import torch as _torch
import torch.nn as _nn
import typing as _t


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Swin Transformer Attention - Source https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
# Note - This is a modified version that is compatible for my use case.
# License - MIT License
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class SwinTransformerAttention(_nn.Module):
    """
    The Swin Transformer attention module that applies self-attention to non-overlapping windows of patch embeddings.
    Attention can either be standard or shifted, where shifted attention is used to attend neighboring patches.
    """

    def __init__(
            self,
            *,
            in_patches: int,
            in_channels: int,
            dropout: bool,
            num_attention_heads: int,
            patch_resolution: _t.Tuple[int, int],
            shifted_window: bool,
            window_size: _t.Tuple[int, int],
    ) -> None:
        """

        :param in_channels: The length of the patch embeddings.
        :param in_patches: The number of patch embeddings.
        :param dropout: If True, apply dropout to the attention weights
        :param num_attention_heads: Number of attention heads.
        :param patch_resolution: The resolution of the image after patching.
        :param shifted_window: If True, use shifted window attention to attend neighboring patches.
        :param window_size: The size of the window used for attention.
        """
        super().__init__()

        self.__in_patches, self.__in_channels = in_patches, in_channels
        self.__patch_resolution = patch_resolution
        self.__window_attention = _WindowAttention(in_channels, window_size, num_attention_heads, dropout)
        self.__window_size = window_size

        # Shifted Window Attention
        attn_mask = None
        shift_size = None
        if shifted_window:
            # calculate attention mask for SW-MSA
            H, W = patch_resolution
            img_mask = _torch.zeros((1, H, W, 1))  # 1 H W 1

            window_size_h, window_size_w = window_size
            shift_size = window_size_h // 2
            h_slices = (slice(0, -window_size_h),
                        slice(-window_size_h, -shift_size),
                        slice(-shift_size, None))
            w_slices = (slice(0, -window_size_w),
                        slice(-window_size_w, -shift_size),
                        slice(-shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = _window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, window_size_h * window_size_w)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        self.__attn_mask = attn_mask
        self.__shift_size = shift_size

    def forward(self, patch_embeddings: _torch.Tensor) -> _torch.Tensor:
        """
        Apply self-attention to non-overlapping windows of patch embeddings.

        :param patch_embeddings: The patch embeddings, shape (B, L, C).
        :return: The attended patch embeddings, shape (B, L, C).
        """
        in_patches, in_channels = self.__in_patches, self.__in_channels
        attn_mask = self.__attn_mask
        H, W = self.__patch_resolution
        shift_size = self.__shift_size
        window_attention = self.__window_attention
        window_size = self.__window_size

        B, L, C = patch_embeddings.shape
        assert L == in_patches and C == in_channels, "Input shape is not valid before `SwinTransformerAttention`."

        patch_embeddings = patch_embeddings.view(B, H, W, C)

        shifted_window_attention = attn_mask is not None
        if shifted_window_attention:
            patch_embeddings = _torch.roll(patch_embeddings, shifts=(-shift_size, -shift_size), dims=(1, 2))

        # Partition the patch embeddings into non-overlapping windows
        patch_embeddings = _window_partition(patch_embeddings, window_size)
        window_size_h, window_size_w = window_size
        patch_embeddings = patch_embeddings.view(-1, window_size_h * window_size_w, C)

        # Apply the window attention mechanism
        patch_embeddings = window_attention(patch_embeddings, mask=attn_mask)
        patch_embeddings = patch_embeddings.view(-1, window_size_h, window_size_w, C)

        # Reconstruct the patch embeddings
        patch_embeddings = _window_reverse(patch_embeddings, window_size, H, W)

        # Reverse Cyclic Shift
        if shifted_window_attention:
            patch_embeddings = _torch.roll(patch_embeddings, shifts=(shift_size, shift_size), dims=(1, 2))

        patch_embeddings = patch_embeddings.view(B, H * W, C)

        assert patch_embeddings.shape[1:] == (in_patches, in_channels) \
            , "Output shape is not valid after `SwinTransformerAttention`."

        return patch_embeddings


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Private Helpers
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class _WindowAttention(_nn.Module):
    """
    The window attention module that applies self-attention to non-overlapping windows of patch embeddings.
    """

    def __init__(self, in_channels: int, window_size: _t.Tuple[int, int], num_attention_heads: int,
                 dropout: bool) -> None:
        """

        :param in_channels: Patch embedding length
        :param window_size: Window size (height, width), patches are contained within each window
        :param num_attention_heads: Number of attention heads
        :param dropout: Whether to apply dropout to the attention weights
        """
        super().__init__()
        head_dim = in_channels // num_attention_heads
        window_size_h, window_size_w = window_size
        dropout_rate = 0.0 if dropout is False else 0.1

        self.__window_size = window_size  # Wh, Ww
        self.__num_heads = num_attention_heads
        self.__scale = head_dim ** -0.5
        self.__qkv = _nn.Linear(in_channels, in_channels * 3, bias=True)
        self.__attn_drop = _nn.Dropout(dropout_rate)
        self.__proj = _nn.Linear(in_channels, in_channels)
        self.__proj_drop = _nn.Dropout(dropout_rate)
        self.__softmax = _nn.Softmax(dim=-1)

        # Define a parameter table of relative position bias
        relative_position_bias_table = _nn.Parameter(
            _torch.zeros((2 * window_size_h - 1) * (2 * window_size_w - 1), num_attention_heads)
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

        :param windowed_patch_embeddings: The windowed patch embeddings, shape (num_windows*B, window_size, in_channels)
        :param mask: The attention mask used for shifted window attention, shape (num_windows*B, window_size, window_size)
        :return: The attended windowed patch embeddings, shape (num_windows*B, window_size, in_channels)
        """
        b, n, c = windowed_patch_embeddings.shape  # num_windows*B, window_size, in_channels
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

        windowed_patch_embeddings = ((attn @ v).transpose(1, 2)
                                     .reshape(b, n, c))  # num_windows*B, window_size, in_channels
        windowed_patch_embeddings = self.__proj(windowed_patch_embeddings)
        windowed_patch_embeddings = self.__proj_drop(windowed_patch_embeddings)

        return windowed_patch_embeddings


def _window_partition(x: _torch.Tensor, window_size: _t.Tuple[int, int]) -> _torch.Tensor:
    """
    Partition the input tensor into non-overlapping windows of size `window_size`.

    :param x: Patch embeddings to partition, shape (B, H, W, C).
    :param window_size: The size of the window used for partitioning.
    :return: Partitioned windows, shape (num_windows*B, window_size, window_size, C).
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)

    return windows


def _window_reverse(x: _torch.Tensor, window_size: _t.Tuple[int, int], H: int, W: int) -> _torch.Tensor:
    """
    Reverse the operation of `_window_partition`.
    
    :param x: Windowed patch embeddings of shape (num_windows*B, window_size, window_size, C). 
    :param window_size: The size of the windows.
    :param H: The height of the image after patching. 
    :param W: The width of the image after patching.
    :return: Non-windowed patch embeddings of shape (B, H, W, C).
    """
    B = int(x.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = x.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
