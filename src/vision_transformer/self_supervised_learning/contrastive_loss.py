import random as _random
import typing as _t

import torch as _torch
import torch.nn as _nn
import torchvision.transforms as _T

import src.vision_transformer.model.base as _base


class ContrastivePreTraining(_nn.Module):
    """
    Contrastive pre-training for semantic segmentation vision transformers.
    """

    def __init__(
            self,
            model: _base.SemanticSegmentationVisionTransformerBase,
            encoder_dims: _t.List[_t.Tuple[int, int]],
            projection_dim: int
    ) -> None:
        """
        Initialize the contrastive pre-training mixin.

        :param model: The model to be trained.
        :param encoder_dims: Dimensions of the encoder's output features.
        :param projection_dim: Dimension of the projection space.
        """
        super(ContrastivePreTraining, self).__init__()
        self.__model = model
        self.__projection_heads = _nn.ModuleList(
            [
                _nn.Sequential(
                    _nn.Linear(encoder_dim, encoder_dim),
                    _nn.BatchNorm1d(num_patches),
                    _nn.ReLU(),
                    _nn.Linear(encoder_dim, projection_dim),
                    _nn.BatchNorm1d(num_patches),
                )
                for num_patches, encoder_dim in encoder_dims
            ]
        )
        self.__transformations = [
            _T.RandomRotation(degrees=90),
            _T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        ]

    def forward(
            self, x: _torch.Tensor, temperature: float = 0.5
    ) -> _torch.Tensor:
        """
        Forward pass of the contrastive pre-training.

        :param x: The input tensor.
        :param temperature: Temperature parameter for scaling the logits.
        :return: The contrastive loss.
        """
        model = self.__model
        projection_heads = self.__projection_heads
        transformations = self.__transformations

        # Apply random transformations
        # - Select two random transformations
        t1 = _random.choice(transformations)
        t2 = _random.choice(transformations)
        # - Apply transformations
        x1 = t1(x)
        x2 = t2(x)

        # Forward pass
        # - Convert images to patch embeddings
        x1 = model.apply_patch_embedding_stage(x1)
        x2 = model.apply_patch_embedding_stage(x2)
        # - Encode patch embeddings
        y1 = model.apply_encoder_stage(**x1)
        y2 = model.apply_encoder_stage(**x2)

        # Apply projection head to each patch embedding scale in the encoder output
        z1 = [
            projection_head(y1_)
            for y1_, projection_head in zip(y1.values(), projection_heads)
        ]
        z2 = [
            projection_head(y2_)
            for y2_, projection_head in zip(y2.values(), projection_heads)
        ]
        # - Reshape the projected embeddings to [B*C, E] from [B, C, E]
        z1 = [z1_.reshape(-1, z1_.shape[-1]) for z1_ in z1]
        z2 = [z2_.reshape(-1, z2_.shape[-1]) for z2_ in z2]

        # Compute the contrastive loss for each projection head
        # - Combine the contrastive loss from each projection head into a single loss tensor
        # - Compute the mean loss over all projection heads
        loss = _torch.stack(
            [
                self.__compute_info_nce_loss(z1_, z2_, temperature)
                for z1_, z2_ in zip(z1, z2)
            ]
        ).mean()

        return loss

    @staticmethod
    def __compute_info_nce_loss(
            z1: _torch.Tensor, z2: _torch.Tensor, temperature: float
    ) -> _torch.Tensor:
        pass
