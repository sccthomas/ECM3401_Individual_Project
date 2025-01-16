import torch as _torch
import torch.nn as _nn


class PatchFusion(_nn.Module):
    """
    Patch Fusion layer.
    - Fuses patches from different scales to a common scale.
    """

    def __init__(self, *, in_patches: int, in_embed: int, out_patches: int, out_embed: int):
        """

        :param in_patches: The number of input patches.
        :param in_embed: The length of the input patch embeddings.
        :param out_patches: The number of output patches.
        :param out_embed: The length of the output patch embeddings.
        """
        super(PatchFusion, self).__init__()

        self.__feature_projector = _nn.Linear(in_embed, out_embed)
        self.__sequence_expander = _nn.ConvTranspose1d(
            in_channels=in_patches, out_channels=out_patches, kernel_size=1
        )
        self.__norm = _nn.LayerNorm(out_embed, eps=1e-6)

    def forward(self, tensor: _torch.Tensor, target_tensor: _torch.Tensor) -> _torch.Tensor:
        """
        Forward pass of the patch fusion layer.

        :param tensor: Tensor to be fused.
        :param target_tensor: Target tensor to be fused with.
        :return: Fused tensor.
        """
        feature_projector = self.__feature_projector
        sequence_expander = self.__sequence_expander
        norm = self.__norm

        tensor = feature_projector(tensor)
        tensor = sequence_expander(tensor)

        tensor = tensor + target_tensor

        tensor = norm(tensor).float()

        return tensor
