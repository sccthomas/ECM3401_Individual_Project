import random as _random
import typing as _t

import matplotlib.pyplot as _plt
import numpy as _np
import sklearn.manifold as _manifold
import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F
import torchvision.transforms.v2 as _T

import src.dataset.snow as _snow
import src.self_supervised_learning.base as _ssl_base
import src.vision_transformer.model as _model


class ContrastivePreTraining(_ssl_base.SelfSupervisedLoss):
    """
    Contrastive Self-Supervised pre-training for semantic segmentation vision transformers.
    """

    def __init__(
            self,
            model: _model.SemanticSegmentationVisionTransformer,
            encoder_dims: _t.List[int],
            projection_dim: int,
            temperature: float = 0.2,
            pooling_method: str = "max",
    ) -> None:
        """

        :param model: The model to be trained.
        :param encoder_dims: Dimensions of the encoder's output features.
        :param projection_dim: Dimension of the projection space.
        :param temperature: Temperature parameter for scaling the logits.
        :param pooling_method: The pooling method to use.
        """
        assert temperature > 0, "Temperature must be positive and non-zero."

        super(ContrastivePreTraining, self).__init__(model=model)
        self.__projection_heads = _nn.ModuleList(
            [
                _ProjectionHead(encoder_dim, projection_dim, pooling_method=pooling_method)
                for encoder_dim in encoder_dims
            ]
        )

        self.__transformations = [
            (_nn.Identity(), _T.RandomVerticalFlip(p=1)),
            (_T.RandomHorizontalFlip(p=1), _nn.Identity()),
            (_T.RandomHorizontalFlip(p=1), _T.RandomVerticalFlip(p=1)),

            (_nn.Identity(), _T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)),
            (_T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), _nn.Identity()),
            (_T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
             _T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)),

            (_nn.Identity(), _T.GaussianBlur(kernel_size=5, sigma=(0.2, 0.5))),
            (_T.GaussianBlur(kernel_size=5, sigma=(0.2, 0.5)), _nn.Identity(),),
            (_T.GaussianBlur(kernel_size=5, sigma=(0.2, 0.5)), _T.GaussianBlur(kernel_size=5, sigma=(0.2, 0.5))),

            (_T.RandomAdjustSharpness(sharpness_factor=0, p=1), _nn.Identity()),
            (_nn.Identity(), _T.RandomAdjustSharpness(sharpness_factor=0, p=1)),
            (_T.RandomAdjustSharpness(sharpness_factor=2, p=1), _nn.Identity()),
            (_nn.Identity(), _T.RandomAdjustSharpness(sharpness_factor=2, p=1)),
            (_T.RandomAdjustSharpness(sharpness_factor=0, p=1), _T.RandomAdjustSharpness(sharpness_factor=2, p=1)),

            (_nn.Identity(), _T.RandomRotation(degrees=360)),
            (_T.RandomRotation(degrees=360), _nn.Identity()),
            (_T.RandomRotation(degrees=360), _T.RandomRotation(degrees=360)),

            (_T.RandomErasing(p=1, scale=(0.02, 0.1)), _nn.Identity()),
            (_nn.Identity(), _T.RandomErasing(p=1, scale=(0.02, 0.1))),
            (_T.RandomErasing(p=1, scale=(0.02, 0.1)), _T.RandomErasing(p=1, scale=(0.02, 0.1))),
        ]
        self.__temperature = temperature
        self.__criterion = _nn.CrossEntropyLoss()

        # Initialize weights
        self.__initialize_weights()

    def forward(self, x: _torch.Tensor) -> _t.Tuple[_torch.Tensor, _torch.Tensor]:
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

        # Apply random transformations
        # - Select two random transformations
        transformation_1, transformation_2 = _random.choice(transformations)

        z = ()
        for transformation_x in [transformation_1, transformation_2]:
            x_ = transformation_x(x)
            x_ = model.apply_patch_embedding_stage(x_)
            x_ = model.apply_encoder_stage(patch_embeddings=x_)
            for key, projection_head in zip(x_.keys(), projection_heads):
                x_[key] = projection_head(x_[key])
            z += (_torch.stack(list(x_.values()), dim=1).sum(dim=1),)

        assert z[0].shape == z[0].shape, "Embeddings must have the same shape."

        return z

    def forward_loss(self, x: _torch.Tensor) -> _torch.Tensor:
        """
        Forward pass for the contrastive pre-training. This method wraps the forward pass and calculates the
        loss between the positive patch embeddings and the negative patch embeddings.

        :param x: The input tensor.
        :return: The loss.
        """
        x1, x2 = self.forward(x)

        # Compute the loss
        loss = self.__loss_fn(x1, x2)

        # Check if the loss is nan
        if _torch.isnan(loss):
            raise ValueError("Loss is NaN.")

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

        assert z1.shape == z2.shape, "Embeddings must have the same shape."

        B = z1.size(0)
        similarity_matrix = _F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1) / temperature
        labels = _torch.arange(B, device=similarity_matrix.device)
        loss = criterion(similarity_matrix, labels)

        # Check if the loss is nan
        if _torch.isnan(loss):
            raise ValueError("Loss is NaN.")

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


def visualize_tsne(
        model: ContrastivePreTraining,
        images: _torch.Tensor,
        title="t-SNE Visualization of Image-Level Embeddings",
        n_components: int = 2,
        perplexity: float = 3,
        normalise: bool = True,
) -> None:
    """
    Visualizes image-level embeddings using t-SNE, where each image is represented as a single point.

    :param model: Contrastive pre-training model.
    :param images: Input images.
    :param title: Title of the plot.
    :param n_components: Number of components for t-SNE.
    :param perplexity: Perplexity parameter for t-SNE.
    :param normalise: Whether to normalize the image.

    """
    images = _T.Normalize(mean=_snow.MEAN, std=_snow.STD)(images) if normalise else images
    z1, z2 = model.forward(images)

    with _torch.no_grad():
        B, E = z1.shape  # Batch, Patches, Embedding Dim
        print(z1.shape)

        # Combine embeddings for t-SNE visualization [2B, E]
        embeddings = _np.concatenate([z1, z2], axis=0)

        # Apply t-SNE
        tsne = _manifold.TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Define a distinct color for each image
        cmap = _plt.get_cmap("tab10")  # "tab10" has 10 distinct colors
        colors = [cmap(i % 10) for i in range(B)]  # Assign each image a unique color

        # Plot
        _plt.figure(figsize=(8, 6))

        for i in range(B):  # Each image
            idx_x1, idx_x2 = i, B + i  # First and second augmentation indices
            color = colors[i]  # Unique color for this image

            # Scatter points
            _plt.scatter(embeddings_2d[idx_x1, 0], embeddings_2d[idx_x1, 1], color=color, label=f'Image {i}',
                         alpha=0.8,
                         marker='o')
            _plt.scatter(embeddings_2d[idx_x2, 0], embeddings_2d[idx_x2, 1], color=color, alpha=0.8, marker='x')

            # Draw line connecting views of the same image
            _plt.plot([embeddings_2d[idx_x1, 0], embeddings_2d[idx_x2, 0]],
                      [embeddings_2d[idx_x1, 1], embeddings_2d[idx_x2, 1]],
                      color=color, alpha=0.5, linestyle="--")

        _plt.legend()
        _plt.title(f'{title} - Combined Patch Embedding Scales')
        _plt.show()


########################################################################################################################
# Private Classes
########################################################################################################################


class _ProjectionHead(_nn.Module):
    """
    Projection head for the contrastive self-supervised learning.

    """

    def __init__(self, input_dim: int, output_dim: int, pooling_method: str = "hybrid") -> None:
        """

        :param input_dim: The input dimension.
        :param output_dim: The output dimension.
        :param pooling_method: The pooling method to use.
        """
        super(_ProjectionHead, self).__init__()
        hidden_dim = (input_dim + output_dim) // 2
        self.__operations = _nn.Sequential(
            _nn.Linear(input_dim, hidden_dim),
            _nn.BatchNorm1d(hidden_dim),
            _nn.ReLU(),
            _nn.Linear(hidden_dim, output_dim),
            _nn.BatchNorm1d(output_dim)
        )
        self.__output_dim = output_dim
        self.__pooling = (
            lambda x: (x.max(dim=1).values + x.mean(dim=1)) / 2
            if pooling_method == "hybrid" else
            (
                x.max(dim=1).values
                if pooling_method == "max" else
                x.mean(dim=1)
            )
        )

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        """
        Forward pass of the projection head.
        :param x: The input tensor. Shape -> [B, P, C]
        :return: The output tensor. Shape -> [B, output_dim]
        """
        operations = self.__operations
        output_dim = self.__output_dim
        pooling = self.__pooling

        B, P, C = x.shape

        # Apply the projection head
        x = x.reshape(B * P, C)
        x = operations(x)
        x = x.reshape(B, P, output_dim)
        # Apply Hybrid Max and Mean Pooling on the patch embeddings
        x = pooling(x)
        x = _F.normalize(x, p=2, dim=-1)

        assert x.shape == (B, output_dim), f"Output shape is incorrect. Expected {(B, output_dim)}, got {x.shape}."

        return x
