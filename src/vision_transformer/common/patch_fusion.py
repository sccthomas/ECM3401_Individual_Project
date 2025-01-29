import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F


class PatchFusion(_nn.Module):
    """
    Patch fusion layer that will fuse the embeddings of the patches to a common scale using learnable and non-learnable
    operations.
    """

    def __init__(self, *, in_embed: int, out_embed: int):
        """

        :param in_embed: The length of the input patch embeddings.
        :param out_embed: The length of the output patch embeddings.
        """
        super(PatchFusion, self).__init__()

        self.__feature_projector = _nn.Linear(in_embed, out_embed)
        self.__norm = _nn.LayerNorm(out_embed, eps=1e-6)

        self.__initialize_weights()

    def forward(self, tensor: _torch.Tensor, target_tensor: _torch.Tensor) -> _torch.Tensor:
        """
        Forward pass of the patch fusion layer.

        :param tensor: Tensor to be fused.
        :param target_tensor: Target tensor to be fused with.
        :return: Fused tensor.
        """
        # attention = self.__attention
        feature_projector = self.__feature_projector
        norm = self.__norm

        # Manipulate the spatial dimension of the tensor to match the target tensor
        P = int(target_tensor.size(1) ** 0.5)
        B, N, C = tensor.shape
        H = W = int(N ** 0.5)
        tensor = tensor.reshape(B, C, H, W)
        # - Interpolate the tensor to the target patch size
        tensor = _F.interpolate(tensor, size=(P, P), mode="bilinear", align_corners=False)
        tensor = tensor.reshape(B, P * P, C)
        # - Project the features to the target feature size
        tensor = feature_projector(tensor)
        tensor = norm(tensor).float()

        # Fuse the tensor with the target tensor
        target_tensor = target_tensor + tensor

        return target_tensor

    def __initialize_weights(self) -> None:
        """
        Initialize the weights of the patch fusion layer.
        """
        feature_projector = self.__feature_projector

        _nn.init.xavier_uniform_(feature_projector.weight)
        _nn.init.constant_(feature_projector.bias, 0)
