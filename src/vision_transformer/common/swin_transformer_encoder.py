import typing as _t

import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F


class SwinTransformerEncoderLayer(_nn.Module):
    """
    Swin Transformer Shifted/Non-Shifted Encoder Layer.
    """

    def __init__(
            self,
            d_model: int,
            nhead: int,
            window_size: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: _t.Union[str, _t.Callable[[_torch.Tensor], _torch.Tensor]] = _F.relu,
            layer_norm_eps: float = 1e-5,
            batch_first: bool = False,
            norm_first: bool = False,
            bias: bool = True,
            use_shifted_windows: bool = False,  # New argument
            device=None,
            dtype=None,
    ) -> None:
        """

        :param d_model: 
        :param nhead:
        :param window_size:
        :param dim_feedforward:
        :param dropout:
        :param activation:
        :param layer_norm_eps:
        :param batch_first:
        :param norm_first:
        :param bias:
        :param use_shifted_windows:
        :param device:
        :param dtype:
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.__window_size = window_size
        self.__use_shifted_windows = use_shifted_windows  # Store shifted windows flag
        self.__self_attn = _nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,
        )

        # Feedforward network
        self.__linear1 = _nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.__dropout = _nn.Dropout(dropout)
        self.__linear2 = _nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.__norm_first = norm_first
        self.__norm1 = _nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.__norm2 = _nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.__dropout1 = _nn.Dropout(dropout)
        self.__dropout2 = _nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _F.relu if activation == "relu" else _F.gelu

        self.__activation = activation

    def forward(
            self,
            src: _torch.Tensor,
            src_mask: _t.Optional[_torch.Tensor] = None,
            src_key_padding_mask: _t.Optional[_torch.Tensor] = None,
            is_causal: bool = False,
    ) -> _torch.Tensor:
        """

        :param src:
        :param src_mask:
        :param src_key_padding_mask:
        :param is_causal:
        :return:
        """
        window_size = self.__window_size
        use_shifted_windows = self.__use_shifted_windows
        self_attn = self.__self_attn
        norm_first = self.__norm_first
        norm1 = self.__norm1
        norm2 = self.__norm2

        batch_size, seq_length, embed_dim = src.shape if self_attn.batch_first else (
            src.shape[1], src.shape[0], src.shape[2])

        # Ensure the sequence length is divisible by the window size
        assert seq_length % window_size == 0, "Sequence length must be divisible by window size."

        # Split into windows
        src_windows = src.unfold(1 if self_attn.batch_first else 0, window_size, window_size)
        src_windows = src_windows.reshape(-1, window_size, embed_dim)  # Flatten windows

        if use_shifted_windows:
            # Apply shift to windows (alternate layers)
            src_windows = self.__apply_shift(src_windows, seq_length)

        # Apply window attention
        attention_output = self.__window_attention(src_windows, src_mask, src_key_padding_mask)

        # Merge windows back
        attention_output = attention_output.reshape(batch_size, -1, embed_dim)  # Combine windows

        # Add & Norm + Feedforward + Add & Norm
        if norm_first:
            x = src + self.__sa_block(norm1(src), src_mask, src_key_padding_mask)
            x = x + self.__ff_block(norm2(x))
        else:
            x = norm1(src + self.__sa_block(src, src_mask, src_key_padding_mask))
            x = norm2(x + self.__ff_block(x))

        return x

    def __window_attention(self, src_windows, src_mask, src_key_padding_mask) -> _torch.Tensor:
        """

        :param src_windows:
        :param src_mask:
        :param src_key_padding_mask:
        :return:
        """
        self_attn = self.__self_attn
        dropout1 = self.__dropout1

        src = self_attn(
            src_windows, src_windows, src_windows,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False
        )[0]
        src = dropout1(src)
        return src

    def __apply_shift(self, src_windows, seq_length) -> _torch.Tensor:
        """

        :param src_windows:
        :param seq_length:
        :return:
        """
        window_size = self.__window_size

        shift_size = window_size // 2
        src_windows = _torch.torch.roll(src_windows, shifts=(-shift_size,), dims=1)
        return src_windows

    def __sa_block(
            self, x: _torch.Tensor, attn_mask: _t.Optional[_torch.Tensor], key_padding_mask: _t.Optional[_torch.Tensor]
    ) -> _torch.Tensor:
        """

        :param x:
        :param attn_mask:
        :param key_padding_mask:
        :return:
        """
        dropout1 = self.__dropout1

        x = self.__window_attention(x, attn_mask, key_padding_mask)
        x = dropout1(x)
        return x

    def __ff_block(self, x: _torch.Tensor) -> _torch.Tensor:
        """

        :param x:
        :return:
        """
        linear1 = self.__linear1
        linear2 = self.__linear2
        dropout = self.__dropout
        dropout2 = self.__dropout2
        activation = self.__activation

        x = linear2(dropout(activation(linear1(x))))
        x = dropout2(x)
        return x
