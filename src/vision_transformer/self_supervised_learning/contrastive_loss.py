import random as _random
import typing as _t

import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F
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
            projection_dim: int,
            temperature: float = 1
    ) -> None:
        """
        Initialize the contrastive pre-training mixin.

        :param model: The model to be trained.
        :param encoder_dims: Dimensions of the encoder's output features.
        :param projection_dim: Dimension of the projection space.
        :param temperature: Temperature parameter for scaling the logits.
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
        self.__temperature = temperature

    @property
    def model(self) -> _base.SemanticSegmentationVisionTransformerBase:
        """
        Get the model being trained.

        :return: The model.
        """
        return self.__model

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        """
        Forward pass of the contrastive pre-training.

        :param x: The input tensor.
        :return: The contrastive loss.
        """
        model = self.__model
        projection_heads = self.__projection_heads
        transformations = self.__transformations
        temperature = self.__temperature

        # Apply random transformations
        # - Select two random transformations
        t1 = _random.choice(transformations)
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
        # - Reshape the projected embeddings to [B*C, E] from [B, C, E]
        x1 = [x1_.reshape(-1, x1_.shape[-1]) for x1_ in x1]
        x2 = [x2_.reshape(-1, x2_.shape[-1]) for x2_ in x2]

        # Compute the contrastive loss for each projection head
        # - Combine the contrastive loss from each projection head into a single loss tensor
        # - Compute the mean loss over all projection heads
        loss = _torch.mean(
            _torch.stack(
                [
                    self.__compute_info_nce_loss(x1_, x2_)
                    for x1_, x2_ in zip(x1, x2)
                ]
            )
        )

        return loss

    def __compute_info_nce_loss(self, embeddings1: _torch.Tensor, embeddings2: _torch.Tensor):
        """
        Compute the InfoNCE loss for a pair of embeddings.

        :param embeddings1: Patch embeddings for scale 1 with on type of augmentation.
        :param embeddings2: Patch embeddings for scale 2 with another type of augmentation.
        :return: The InfoNCE loss.
        """
        temperature = self.__temperature

        # Normalize the embeddings
        embeddings1 = _F.normalize(embeddings1, dim=1)
        embeddings2 = _F.normalize(embeddings2, dim=1)

        # Concatenate the embeddings to form the positive pair (2N, D)
        embeddings = _torch.cat([embeddings1, embeddings2], dim=0)  # Shape: (2N, D)

        # Compute the similarity matrix (2N, 2N)
        similarity_matrix = _torch.mm(embeddings, embeddings.t())  # (2N, 2N)

        # Create labels for positive pairs (i.e., first N are positive to second N)
        N = embeddings1.size(0)
        labels = _torch.arange(N, dtype=_torch.long, device=embeddings.device)
        labels = _torch.cat([labels, labels], dim=0)  # Shape: (2N,)

        # Mask out self-similarities to avoid trivial positives
        mask = _torch.eye(2 * N, device=embeddings.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

        # Apply temperature scaling to similarity matrix
        similarity_matrix /= temperature

        # Clip similarity values to prevent overflow/underflow
        similarity_matrix = _torch.clamp(similarity_matrix, min=-10.0, max=10.0)

        # Subtract the max value from each row for numerically stable softmax
        similarity_matrix = similarity_matrix - similarity_matrix.max(dim=1, keepdim=True)[0]

        # Compute the cross-entropy loss (InfoNCE) using the stable similarity matrix
        loss = _F.cross_entropy(similarity_matrix, labels)

        return loss
