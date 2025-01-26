import random as _random
import typing as _t

import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F
import torchvision.transforms as _T

import src.vision_transformer.model.base as _base
import src.vision_transformer.self_supervised_learning.base as _ssl_base


class ContrastivePreTraining(_ssl_base.SelfSupervisedLoss):
    """
    Contrastive pre-training for semantic segmentation vision transformers.
    """

    def __init__(
            self,
            model: _base.SemanticSegmentationVisionTransformerBase,
            encoder_dims: _t.List[int],
            projection_dim: int,
            temperature: float = 0.5
    ) -> None:
        """
        Initialize the contrastive pre-training mixin.

        :param model: The model to be trained.
        :param encoder_dims: Dimensions of the encoder's output features.
        :param projection_dim: Dimension of the projection space.
        :param temperature: Temperature parameter for scaling the logits.
        """
        assert temperature > 0, "Temperature must be positive and non-zero."

        super(ContrastivePreTraining, self).__init__(model=model)
        self.__projection_heads = _nn.ModuleList(
            [
                _nn.Sequential(
                    _nn.Linear(encoder_dim, hidden_dim),
                    _nn.LayerNorm(hidden_dim),
                    _nn.ReLU(),
                    _nn.Linear(hidden_dim, projection_dim),
                    _nn.LayerNorm(projection_dim),
                )
                for encoder_dim in encoder_dims
                if (hidden_dim := encoder_dim // 2) > projection_dim
            ]
        )
        self.__transformations = [
            _T.RandomRotation(degrees=90),
            _T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        ]
        self.__temperature = temperature

        # Initialize weights
        self.__initialize_weights()

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        """
        Forward pass of the contrastive pre-training.

        :param x: The input tensor.
        :return: The contrastive loss.
        """
        model = self.model
        projection_heads = self.__projection_heads
        transformations = self.__transformations
        temperature = self.__temperature

        # Apply random transformations
        # - Select two random transformations
        t1 = _random.choice(transformations)
        t2 = _random.choice(transformations)
        while t1 == t2:
            t2 = _random.choice(transformations)

        # - Apply transformations
        x1 = t1(x)
        x2 = t2(x)

        # Forward pass
        # - Convert images to patch embeddings
        x1 = model.apply_patch_embedding_stage(x1)  # Output -> dict[str, _torch.Tensor]
        x2 = model.apply_patch_embedding_stage(x2)  # Output -> dict[str, _torch.Tensor]
        # - Encode patch embeddings
        x1 = model.apply_encoder_stage(**x1)  # Output -> dict[str, _torch.Tensor]
        x2 = model.apply_encoder_stage(**x2)  # Output -> dict[str, _torch.Tensor]

        # Apply projection head to each patch embedding scale in the encoder output
        x1 = [
            projection_head(x1_)
            for x1_, projection_head in zip(x1.values(), projection_heads)
        ]
        x2 = [
            projection_head(x2_)
            for x2_, projection_head in zip(x2.values(), projection_heads)
        ]
        # - Reshape the projected embeddings to [B, C * E] from [B, C, E]
        x1 = [x1_.reshape(x1_.shape[0], -1) for x1_ in x1]
        x2 = [x2_.reshape(x2_.shape[0], -1) for x2_ in x2]

        # Compute the contrastive loss
        loss = []
        for x1_, x2_ in zip(x1, x2):
            features = _torch.cat([x1_, x2_], dim=0)
            scale_loss = self.__loss_fn(features)
            loss.append(scale_loss)
        loss = _torch.stack(loss).mean()

        return loss

    def __loss_fn(self, features: _torch.Tensor):
        """
        Compute the InfoNCE loss for a pair of embeddings in a memory-efficient manner.

        :param features: The features to compute the loss for.
        :return: The InfoNCE loss.
        """
        temperature = self.__temperature

        # Normalize features to have unit norm
        features = _F.normalize(features, dim=1)

        # Compute similarity matrix
        similarity_matrix = _torch.matmul(features, features.T) / temperature

        # Get batch size
        batch_size = features.shape[0] // 2

        # Construct labels where each sample's positive pair is in the other view
        labels = _torch.arange(batch_size, device=features.device)
        labels = _torch.cat([labels + batch_size, labels], dim=0)

        # Mask out self-similarities by setting the diagonal elements to -inf
        mask = _torch.eye(2 * batch_size, dtype=_torch.bool, device=features.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))

        # InfoNCE loss
        loss = _F.cross_entropy(similarity_matrix, labels)

        return loss

    def __initialize_weights(self) -> None:
        """
        Initialize the weights of the contrastive loss module.
        """
        projection_heads = self.__projection_heads

        for projection_head in projection_heads:
            for layer in projection_head.modules():
                if isinstance(layer, _nn.Linear):
                    _nn.init.kaiming_normal_(layer.weight)
                    _nn.init.zeros_(layer.bias)
                elif isinstance(layer, _nn.LayerNorm):
                    _nn.init.ones_(layer.weight)
                    _nn.init.zeros_(layer.bias)
