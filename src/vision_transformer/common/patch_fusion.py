import abc as _abc
import typing as _t

import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F


class PatchFusion(_nn.Module):
    """
    Base class for the patch fusion layer.
    """

    def __init__(
            self,
            *,
            patch_embedding_projectors: _nn.ModuleList,
            out_patches: int,
            out_embed: int,
            out_resolution: int,
            in_resolutions: _t.List[int],
            dropout_rate: float,
            use_gated_attention: bool = False,
            num_heads: int = None,
    ) -> None:
        """

        :param patch_embedding_projectors: The patch embedding projectors.
        :param out_patches: The number of output patches.
        :param out_embed: The length of the output patch embeddings.
        :param out_resolution: The resolution of the output patches.
        :param in_resolutions: The resolutions of the input patches.
        :param use_gated_attention: Whether to use gated attention.
        :param num_heads: The number of attention heads. Required if `use_gated_attention` is True.
        """
        super(PatchFusion, self).__init__()

        if use_gated_attention and num_heads is None:
            raise ValueError("The number of attention heads must be provided if using gated attention.")

        self._patch_embedding_projectors = patch_embedding_projectors
        self._gated_attention = (
            _GatedSelfAttention(embed_dim=out_embed, num_heads=num_heads, dropout_rate=dropout_rate)
            if use_gated_attention else None
        )
        self._out_resolution = out_resolution
        self._in_resolutions = in_resolutions
        self._out_patches = out_patches
        self._out_embed = out_embed
        self.__initialize_weights()

    @property
    def attention_scores(self) -> _t.Optional[_torch.Tensor]:
        """
        Get the attention scores of the gated self-attention mechanisms.

        :return: The attention scores.
        """
        gated_attention = self._gated_attention

        attention_scores = gated_attention.attention_scores if gated_attention else None

        return attention_scores

    def forward(
            self, target_tensor: _torch.Tensor, tensors: _t.List[_torch.Tensor], keep_attention_scores: bool = False
    ) -> _torch.Tensor:
        """
        Forward pass of the patch fusion layer to merge together a given tensor with a target tensor by modifying
        spatial dimension using learnable operations.

        :param target_tensor: Target tensor to be fused with. Shape (batch_size, out_patches, out_embed).
        :param tensors: List of tensors to be fused. Shape (batch_size, in_patches, in_embed).
        :param keep_attention_scores: Whether to store the attention scores.
        :return: Fused tensor. Shape (batch_size, out_patches, out_embed).
        """
        patch_embedding_projectors = self._patch_embedding_projectors
        gated_attention = self._gated_attention
        in_resolutions = self._in_resolutions
        out_resolution = self._out_resolution
        out_patches = self._out_patches
        out_embed = self._out_embed

        # Reshape the target tensor to (B, E, P, P)
        B, P, E = target_tensor.shape
        target_tensor = target_tensor.reshape(B, out_resolution, out_resolution, E).permute(0, 3, 1, 2).contiguous()

        # Apply the patch embedding projectors
        tensors_ = _torch.zeros_like(target_tensor)
        for tensor, projector, in_resolution in zip(
                tensors, patch_embedding_projectors, in_resolutions
        ):
            B, _, E = tensor.shape
            tensor = tensor.reshape(B, in_resolution, in_resolution, E).permute(0, 3, 1, 2).contiguous()
            tensors_ = tensors_ + projector(tensor)

        # If using gated attention, apply the gated self-attention mechanism
        if gated_attention:
            target_tensor = gated_attention(target_tensor, keep_attention_scores=keep_attention_scores)

        # Add the target tensor to the fused tensor
        target_tensor = target_tensor + tensors_
        target_tensor = target_tensor.permute(0, 3, 1, 2).reshape(B, out_patches, out_embed).float()

        assert target_tensor.shape == (B, out_patches, out_embed)

        return target_tensor

    @_abc.abstractmethod
    def __initialize_weights(self) -> None:
        """
        Initialize the weights of the patch fusion layer.
        """


class PatchFusionLearnable(PatchFusion):
    """
    Patch fusion layer that will fuse the embeddings of the patches to a common scale using learnable operations as its
    foundations.
    """

    def __init__(
            self,
            *,
            in_dims: _t.List[_t.List[int]],
            out_patches: int,
            out_embed: int,
            dropout_rate: float,
            use_gated_attention: bool = False,
            num_heads: int = None,
    ) -> None:
        """

        :param in_dims: The dimensions of the input tensors.
        :param out_patches: The number of output patches.
        :param out_embed: The length of the output patch embeddings.
        :param dropout_rate: Dropout rate.
        :param use_gated_attention: Whether to use gated attention.
        :param num_heads: The number of attention heads. Required if `use_gated_attention` is True.
        """
        out_resolution = int(out_patches ** 0.5)
        in_resolutions = []
        patch_embedding_projectors = _nn.ModuleList()
        for in_patches, in_embed in in_dims:
            in_resolution = int(in_patches ** 0.5)
            if in_patches < out_patches:
                scale = out_resolution // in_resolution
                operation = _nn.Sequential(
                    _nn.Conv2d(
                        in_channels=in_embed, out_channels=out_embed, kernel_size=1, stride=1
                    ),
                    _nn.Upsample(scale_factor=scale, mode="nearest")
                )
            elif in_patches > out_patches:
                scale = in_resolution // out_resolution
                operation = _nn.Conv2d(
                    in_channels=in_embed, out_channels=out_embed, kernel_size=scale, stride=scale
                )
            else:
                operation = _nn.Identity()

            patch_embedding_projectors.append(
                _nn.Sequential(operation, _nn.BatchNorm2d(out_embed), _nn.ReLU(), _nn.Dropout(dropout_rate))
            )
            in_resolutions.append(in_resolution)

        super(PatchFusionLearnable, self).__init__(
            patch_embedding_projectors=patch_embedding_projectors,
            out_patches=out_patches,
            out_embed=out_embed,
            out_resolution=out_resolution,
            in_resolutions=in_resolutions,
            use_gated_attention=use_gated_attention,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )

        self.__patch_embedding_projectors = patch_embedding_projectors
        self.__out_resolution = out_resolution
        self.__in_resolutions = in_resolutions
        self.__out_patches = out_patches
        self.__out_embed = out_embed

    def __initialize_weights(self) -> None:
        """
        Initialize the weights of the patch fusion layer.
        """
        patch_embedding_projectors = self.__patch_embedding_projectors

        for projector in patch_embedding_projectors:
            for module in projector.modules():
                if isinstance(module, _nn.Conv2d):
                    _nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        _nn.init.constant_(module.bias, 0)
                elif isinstance(module, _nn.BatchNorm2d):
                    _nn.init.constant_(module.weight, 1)
                    _nn.init.constant_(module.bias, 0)


class PatchFusionNonLearnable(PatchFusion):
    """
    Patch fusion layer that will fuse the embeddings of the patches to a common scale using non-learnable operations as
    its foundations.
    """

    def __init__(
            self,
            *,
            in_dims: _t.List[_t.List[int]],
            out_patches: int,
            out_embed: int,
            dropout_rate: float,
            use_gated_attention: bool = False,
            num_heads: int = None,
    ) -> None:
        """

        :param in_dims: The dimensions of the input tensors.
        :param out_patches: The number of output patches.
        :param out_embed: The length of the output patch embeddings.
        :param dropout_rate: Dropout rate.
        :param use_gated_attention: Whether to use gated attention.
        :param num_heads: The number of attention heads. Required if `use_gated_attention` is True.
        """
        out_resolution = int(out_patches ** 0.5)
        in_resolutions = []
        patch_embedding_projectors = _nn.ModuleList()
        for in_patches, in_embed in in_dims:
            in_resolution = int(in_patches ** 0.5)
            operation = _nn.Sequential(
                ModifyEmbeddingDim(in_embed=in_embed, out_embed=out_embed)
                if in_embed != out_embed else
                _nn.Identity()
            )
            if in_patches < out_patches:
                scale = out_resolution // in_resolution
                operation.append(
                    _nn.Upsample(scale_factor=scale, mode="nearest")
                )
            elif in_patches > out_patches:
                scale = in_resolution // out_resolution
                operation.append(
                    _nn.AvgPool2d(kernel_size=scale, stride=scale)
                )

            patch_embedding_projectors.append(
                _nn.Sequential(operation, _nn.BatchNorm2d(out_embed), _nn.ReLU(), _nn.Dropout(dropout_rate))
            )
            in_resolutions.append(in_resolution)

        super(PatchFusionNonLearnable, self).__init__(
            patch_embedding_projectors=patch_embedding_projectors,
            out_patches=out_patches,
            out_embed=out_embed,
            out_resolution=out_resolution,
            in_resolutions=in_resolutions,
            use_gated_attention=use_gated_attention,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )

        self.__patch_embedding_projectors = patch_embedding_projectors
        self.__out_resolution = out_resolution
        self.__in_resolutions = in_resolutions
        self.__out_patches = out_patches
        self.__out_embed = out_embed
        self.__initialize_weights()

    def __initialize_weights(self) -> None:
        """
        Initialize the weights of the patch fusion layer.
        """


####################################################################################################
# Private Helper Functions
####################################################################################################


class _GatedSelfAttention(_nn.Module):
    """
    Gated self-attention mechanism that can be used to fuse the embeddings of the patches.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float) -> None:
        """
        :param embed_dim: The dimension of the input embeddings.
        :param num_heads: The number of attention heads.
        :param dropout_rate: Dropout rate.
        """
        super(_GatedSelfAttention, self).__init__()

        self.__attn = _nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.__gate = _nn.Sequential(
            _nn.Linear(embed_dim, embed_dim),
            _nn.Sigmoid(),
        )
        self.__attention_scores = None

        self.__initialize_weights()

    @property
    def attention_scores(self) -> _t.Optional[_torch.Tensor]:
        """
        Get the attention scores.

        :return: The attention scores.
        """
        return self.__attention_scores

    def forward(self, x: _torch.Tensor, keep_attention_scores: bool = False) -> _torch.Tensor:
        """
        Forward pass of the gated self-attention mechanism.

        :param x: Input tensor. Shape (B, E, H, W).
        :param keep_attention_scores: Whether to store the attention scores.
        :return: Output tensor. Shape (B, E, H, W).
        """
        attn = self.__attn
        gate = self.__gate

        # Reshape the input tensor to (B, H * W, E)
        B, E, H, W = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, H * W, E)

        if keep_attention_scores:
            attn_output, self.__attention_scores = attn(x, x, x, need_weights=True, average_attn_weights=False)
        else:
            attn_output, _ = attn(x, x, x, need_weights=False)
        gate_output = gate(x)
        output = gate_output * attn_output + (1 - gate_output) * x

        # Reshape the output tensor to (B, E, H, W)
        output = output.reshape(B, H, W, E).permute(0, 3, 1, 2).contiguous()

        assert output.shape == (B, E, H, W)

        return output

    def __initialize_weights(self) -> None:
        """
        Initialize the weights of the gated self-attention mechanism.
        """
        attn = self.__attn
        gate = self.__gate

        for module in attn.modules():
            if isinstance(module, _nn.Linear):
                _nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    _nn.init.constant_(module.bias, 0)

        for module in gate.modules():
            if isinstance(module, _nn.Linear):
                _nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    _nn.init.constant_(module.bias, 0)


class ModifyEmbeddingDim(_nn.Module):
    """
    Modify the embedding dimension of the input tensor.
    """

    def __init__(self, in_embed: int, out_embed: int):
        """

        :param in_embed: The input embedding dimension.
        :param out_embed: The output embedding dimension.
        """
        super(ModifyEmbeddingDim, self).__init__()

        self.__in_embed = in_embed
        self.__out_embed = out_embed
        self.__increase = True if in_embed < out_embed else False

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        in_embed = self.__in_embed
        out_embed = self.__out_embed
        increase = self.__increase

        if increase:
            x = _torch.cat([x, x[:, :out_embed - in_embed, :, :]], dim=1)
        else:
            x = _F.adaptive_avg_pool2d(
                x.permute(0, 2, 3, 1), (x.size(-1), out_embed)
            ).permute(0, 3, 1, 2)

        return x
