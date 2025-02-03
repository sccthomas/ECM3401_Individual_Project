import torch as _torch
import torch.nn as _nn


class PatchFusion(_nn.Module):
    """
    Patch fusion layer that will fuse the embeddings of the patches to a common scale.
    """

    def __init__(
            self, *, in_patches: int, in_embed: int, out_patches: int, out_embed: int, dropout_rate: float
    ) -> None:
        """

        :param in_patches: The number of input patches.
        :param in_embed: The length of the input patch embeddings.
        :param out_patches: The number of output patches.
        :param out_embed: The length of the output patch embeddings.
        :param dropout_rate: Dropout rate.
        """
        super(PatchFusion, self).__init__()

        in_resolution = int(in_patches ** 0.5)
        out_resolution = int(out_patches ** 0.5)

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

        self.__patch_embedding_projector = _nn.Sequential(
            operation,
            _nn.BatchNorm2d(out_embed),
            _nn.ReLU(),
            _nn.Dropout(dropout_rate)
        )
        self.__in_resolution = in_resolution
        self.__out_patches = out_patches
        self.__out_embed = out_embed
        self.__initialize_weights()

    def forward(self, tensor: _torch.Tensor, target_tensor: _torch.Tensor) -> _torch.Tensor:
        """
        Forward pass of the patch fusion layer to merge together a given tensor with a target tensor by modifying
        spatial dimension using learnable operations.

        :param tensor: Tensor to be fused. Shape (batch_size, in_patches, in_embed).
        :param target_tensor: Target tensor to be fused with. Shape (batch_size, out_patches, out_embed).
        :return: Fused tensor. Shape (batch_size, out_patches, out_embed).
        """
        patch_embedding_projector = self.__patch_embedding_projector
        in_resolution = self.__in_resolution
        out_patches = self.__out_patches
        out_embed = self.__out_embed

        B, P, E = tensor.shape
        tensor = tensor.reshape(B, E, in_resolution, in_resolution)
        tensor = patch_embedding_projector(tensor)
        tensor = tensor.reshape(B, out_patches, out_embed)

        target_tensor = (target_tensor + tensor).float()

        return target_tensor

    def __initialize_weights(self) -> None:
        """
        Initialize the weights of the patch fusion layer.
        """
        patch_embedding_projector = self.__patch_embedding_projector

        for module in patch_embedding_projector:
            if isinstance(module, _nn.Conv2d) or isinstance(module, _nn.ConvTranspose2d):
                _nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                _nn.init.zeros_(module.bias)
            elif isinstance(module, _nn.BatchNorm2d):
                _nn.init.ones_(module.weight)
                _nn.init.zeros_(module.bias)
