import random as _random
import typing as _t

import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F
import torchvision.transforms as _T

import src.self_supervised_learning.base as _ssl_base
import src.vision_transformer.model.base as _base


class ContrastivePreTraining(_ssl_base.SelfSupervisedLoss):
    """
    Contrastive Self-Supervised pre-training for semantic segmentation vision transformers.
    """

    def __init__(
            self,
            model: _base.SemanticSegmentationVisionTransformerBase,
            encoder_dims: _t.List[int],
            projection_dim: int,
            temperature: float = 0.1
    ) -> None:
        """

        :param model: The model to be trained.
        :param encoder_dims: Dimensions of the encoder's output features.
        :param projection_dim: Dimension of the projection space.
        :param temperature: Temperature parameter for scaling the logits.
        """
        assert temperature > 0, "Temperature must be positive and non-zero."

        super(ContrastivePreTraining, self).__init__(model=model)
        self.__projection_heads = _nn.ModuleList(
            [
                _ProjectionHead(encoder_dim, hidden_dim, projection_dim)
                for encoder_dim in encoder_dims
                if (hidden_dim := encoder_dim // 2) > projection_dim
            ]
        )
        self.__transformations = [
            _T.RandomRotation(degrees=90),
            _T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        ]
        self.__temperature = temperature
        self.__criterion = _nn.CrossEntropyLoss()

        # Initialize weights
        self.__initialize_weights()

    def forward(self, x: _torch.Tensor) -> _t.Tuple[_t.List[_torch.Tensor], _t.List[_torch.Tensor]]:
        """
        Forward pass of the contrastive pre-training. This method applied 2 random transformations to the input tensor
        which represent positive pairs. The model encoder is then applied to the transformed tensors and the contrastive
        loss is computed between the positive patch embeddings pairs and the negative patch embedding pairs which are
        all other images which have been transformed. The loss is aimed at maximizing the similarity between the
        positive pairs which are the same image and minimizing the similarity between the negative pairs which are
        different images (2N -1).

        :param x: The input tensor.
        :return: The positive and negative patch embeddings.
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

        return x1, x2

    def forward_loss(self, x: _torch.Tensor) -> _torch.Tensor:
        """
        Forward pass for the contrastive pre-training. This method wraps the forward pass and calculates the
        loss between the positive patch embeddings and the negative patch embeddings.

        :param x: The input tensor.
        :return: The loss.
        """
        x1, x2 = self.forward(x)

        # Compute the contrastive loss
        loss = [
            self.__loss_fn(x1_, x2_)
            for x1_, x2_ in zip(x1, x2)
        ]
        loss = _torch.stack(loss).mean()

        return loss

    def __loss_fn(self, z1: _torch.Tensor, z2: _torch.Tensor):
        """
        Compute the InfoNCE loss.

        :param z1: The first set of embeddings.
        :param z2: The second set of embeddings.
        :return: The InfoNCE loss.
        """
        temperature = self.__temperature
        criterion = self.__criterion

        B, P, _ = z1.shape
        z1 = _F.normalize(z1, dim=-1)  # Normalize along the embedding dimension
        z2 = _F.normalize(z2, dim=-1)

        # Compute similarity matrix
        z1_flat = z1.view(B * P, -1)
        z2_flat = z2.view(B * P, -1)
        similarity_matrix = _torch.mm(z1_flat, z2_flat.t()) / temperature

        # Labels for contrastive loss
        labels = _torch.arange(B * P).to(z1.device)

        # Loss for z1 -> z2 and z2 -> z1
        loss_1 = criterion(similarity_matrix, labels)
        loss_2 = criterion(similarity_matrix.t(), labels)

        # Average the losses
        loss = (loss_1 + loss_2) / 2
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


########################################################################################################################
# Private Classes
########################################################################################################################


class _ProjectionHead(_nn.Module):
    """
    Projection head for the contrastive self-supervised learning.

    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """

        :param input_dim: The input dimension.
        :param hidden_dim: The hidden dimension.
        :param output_dim: The output dimension.
        """
        super(_ProjectionHead, self).__init__()
        self.__fc1 = _nn.Linear(input_dim, hidden_dim)
        self.__bn1 = _nn.BatchNorm1d(hidden_dim)
        self.__fc2 = _nn.Linear(hidden_dim, output_dim)
        self.__bn2 = _nn.BatchNorm1d(output_dim)

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        """
        Forward pass of the projection head.
        :param x: The input tensor.
        :return: The output tensor.
        """
        fc1 = self.__fc1
        bn1 = self.__bn1
        fc2 = self.__fc2
        bn2 = self.__bn2

        # x shape: [B, P, C]
        B, P, C = x.shape
        x = x.view(-1, C)  # Flatten patches into batch dimension
        x = fc1(x)
        x = _F.relu(bn1(x))
        x = fc2(x)
        x = bn2(x)
        x = x.view(B, P, -1)  # Reshape back to [B, P, output_dim]

        return x
