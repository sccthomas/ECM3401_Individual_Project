import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F


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
        self.__post_interpolation_norm = _nn.LayerNorm(out_embed, eps=1e-6)
        self.__fusion_layer = _nn.Linear(out_embed * 2, out_embed)
        self.__post_fusion_norm = _nn.LayerNorm(out_embed, eps=1e-6)

        self.__initialize_weights()

    def forward(self, tensor: _torch.Tensor, target_tensor: _torch.Tensor) -> _torch.Tensor:
        """
        Forward pass of the patch fusion layer.

        :param tensor: Tensor to be fused.
        :param target_tensor: Target tensor to be fused with.
        :return: Fused tensor.
        """
        feature_projector = self.__feature_projector
        fusion_layer = self.__fusion_layer
        post_interpolation_norm = self.__post_interpolation_norm
        post_fusion_norm = self.__post_fusion_norm

        P = target_tensor.size(1)
        tensor = feature_projector(tensor).permute(0, 2, 1)
        tensor = _F.interpolate(tensor, size=(P,), mode="nearest").permute(0, 2, 1)
        tensor = post_interpolation_norm(tensor).float()

        # Fusion (Concatenation + Learnable Fusion)
        fused = _torch.cat([tensor, target_tensor], dim=-1)  # Concatenate embeddings
        fused = fusion_layer(fused)  # Learnable fusion layer
        fused = post_fusion_norm(fused).float()

        return fused

    def __initialize_weights(self) -> None:
        """
        Initialize the weights of the patch fusion layer.
        """
        feature_projector = self.__feature_projector
        fusion_layer = self.__fusion_layer

        _nn.init.xavier_uniform_(feature_projector.weight)
        _nn.init.constant_(feature_projector.bias, 0)
        _nn.init.xavier_uniform_(fusion_layer.weight)
        _nn.init.constant_(fusion_layer.bias, 0)
