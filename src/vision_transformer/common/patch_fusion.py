import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F


class PatchFusionLearnable(_nn.Module):
    """
    Patch fusion layer that will fuse the embeddings of the patches to a common scale using learnable operations.
    """

    def __init__(self, *, in_patches: int, in_embed: int, out_patches: int, out_embed: int):
        """

        :param in_patches: The number of input patches.
        :param in_embed: The length of the input patch embeddings.
        :param out_patches: The number of output patches.
        :param out_embed: The length of the output patch embeddings.
        """
        super(PatchFusionLearnable, self).__init__()

        self.__feature_projector = _nn.Linear(in_embed, out_embed)
        self.__sequence_projector = (
            _nn.ConvTranspose1d(
                in_channels=in_patches, out_channels=out_patches, kernel_size=1
            )
            if in_patches < out_patches else
            _nn.Conv1d(
                in_channels=in_patches, out_channels=out_patches, kernel_size=1
            )
            if in_patches > out_patches else
            _nn.Identity()
        )
        self.__norm = _nn.LayerNorm(out_embed, eps=1e-6)

        self.__initialize_weights()

    def forward(self, tensor: _torch.Tensor, target_tensor: _torch.Tensor) -> _torch.Tensor:
        """
        Forward pass of the patch fusion layer to merge together a given tensor with a target tensor by modifying
        spatial dimension using learnable operations.

        :param tensor: Tensor to be fused. Shape (batch_size, in_patches, in_embed).
        :param target_tensor: Target tensor to be fused with. Shape (batch_size, out_patches, out_embed).
        :return: Fused tensor. Shape (batch_size, out_patches, out_embed).
        """
        feature_projector = self.__feature_projector
        sequence_projector = self.__sequence_projector
        norm = self.__norm

        tensor = feature_projector(tensor)
        tensor = sequence_projector(tensor)

        tensor = tensor + target_tensor

        tensor = norm(tensor).float()

        return tensor

    def __initialize_weights(self) -> None:
        """
        Initialize the weights of the patch fusion layer.
        """
        feature_projector = self.__feature_projector
        sequence_projector = self.__sequence_projector

        _nn.init.xavier_uniform_(feature_projector.weight)
        _nn.init.constant_(feature_projector.bias, 0)

        _nn.init.xavier_uniform_(sequence_projector.weight)
        _nn.init.constant_(sequence_projector.bias, 0)


class PatchFusionNonLearnable(_nn.Module):
    """
    Patch fusion layer that will fuse the embeddings of the patches to a common scale using learnable and non-learnable
    operations.
    """

    def __init__(self, *, in_embed: int, out_embed: int):
        """

        :param in_embed: The length of the input patch embeddings.
        :param out_embed: The length of the output patch embeddings.
        """
        super(PatchFusionNonLearnable, self).__init__()

        self.__attention = _nn.MultiheadAttention(embed_dim=out_embed, num_heads=4, batch_first=True)
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
        attention = self.__attention
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
        # - Apply cross multi-head attention
        tensor, _ = attention(query=target_tensor, key=tensor, value=tensor)

        # Fuse the tensor with the target tensor
        tensor = tensor + target_tensor
        tensor = norm(tensor).float()

        return tensor

    def __initialize_weights(self) -> None:
        """
        Initialize the weights of the patch fusion layer.
        """
        feature_projector = self.__feature_projector

        _nn.init.xavier_uniform_(feature_projector.weight)
        _nn.init.constant_(feature_projector.bias, 0)
