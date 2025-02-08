import random as _random
import typing as _t

import matplotlib.pyplot as _plt
import numpy as _np
import sklearn.manifold as _manifold
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
            temperature: float = 0.2
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
                _ProjectionHead(encoder_dim, projection_dim)
                for encoder_dim in encoder_dims
            ]
        )
        self.__transformations = [
            _T.RandomRotation(degrees=90),
            _T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            _T.RandomHorizontalFlip(),
            _T.RandomVerticalFlip(),
            _T.RandomAdjustSharpness(0),
            _T.RandomAdjustSharpness(2),
            _T.RandomErasing(),
            _T.RandomInvert(),
        ]
        self.__temperature = temperature
        self.__criterion = _nn.CrossEntropyLoss()

        # Initialize weights
        self.__initialize_weights()

    def forward(self, x: _torch.Tensor) -> _t.Tuple[_t.Dict[str, _torch.Tensor], _t.Dict[str, _torch.Tensor]]:
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
        t1 = _random.choice(transformations)
        t2 = _random.choice(transformations)
        while t1 == t2:
            t2 = _random.choice(transformations)

        # - Apply transformations
        x1 = t1(x)
        x2 = t2(x)
        del x

        # Forward pass
        # - Convert images to patch embeddings
        x1 = model.apply_patch_embedding_stage(x1)  # Output -> dict[str, _torch.Tensor]
        x2 = model.apply_patch_embedding_stage(x2)  # Output -> dict[str, _torch.Tensor]
        # - Encode patch embeddings
        x1, _ = model.apply_encoder_stage(patch_embeddings=x1)  # Output -> dict[str, _torch.Tensor]
        x2, _ = model.apply_encoder_stage(patch_embeddings=x2)  # Output -> dict[str, _torch.Tensor]

        # Apply projection head to each patch embedding scale in the encoder output
        keys = x1.keys()
        assert keys == x2.keys(), "Scale names do not match."
        for key, projection_head in zip(keys, projection_heads):
            x1[key] = projection_head(x1[key])
            x2[key] = projection_head(x2[key])

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
            self.__loss_fn(z1, z2)
            for (key1, z1), (key2, z2) in zip(x1.items(), x2.items())
            if key1 == key2
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
        z1 = z1.view(B * P, -1)
        z2 = z2.view(B * P, -1)
        similarity_matrix = _torch.mm(z1, z2.t()) / temperature
        del z1, z2

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


def visualize_tsne(
        model: ContrastivePreTraining,
        images: _torch.Tensor,
        title="t-SNE Visualization of Image-Level Embeddings",
) -> None:
    """
    Visualizes image-level embeddings using t-SNE, where each image is represented as a single point.

    :param model: Contrastive pre-training model.
    :param images: Input images.
    :param title: Title of the plot.
    """
    z1, z2 = model.forward(images)

    scales = z1.keys()
    assert scales == z2.keys(), "Scale names do not match."

    with _torch.no_grad():
        for scale in scales:
            x1 = z1[scale]
            x2 = z2[scale]

            B, P, E = x1.shape  # Batch, Patches, Embedding Dim

            # Compute mean patch embedding per image [B, E]
            x1_avg = x1.mean(dim=1).cpu().numpy()  # Averaging over patches
            x2_avg = x2.mean(dim=1).cpu().numpy()

            # Combine embeddings for t-SNE visualization [2B, E]
            embeddings = _np.concatenate([x1_avg, x2_avg], axis=0)

            # Create labels: Each image gets a unique label
            labels = _np.concatenate([_np.arange(B), _np.arange(B)])  # Shape: [2B]

            # Apply t-SNE
            tsne = _manifold.TSNE(n_components=2, perplexity=10, random_state=42)
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
            _plt.title(f'{title}_{scale}')
            _plt.show()


########################################################################################################################
# Private Classes
########################################################################################################################


class _ProjectionHead(_nn.Module):
    """
    Projection head for the contrastive self-supervised learning.

    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """

        :param input_dim: The input dimension.
        :param output_dim: The output dimension.
        """
        super(_ProjectionHead, self).__init__()
        hidden_dim = input_dim // 2
        self.__fc1 = _nn.Linear(input_dim, hidden_dim)
        self.__n1 = _nn.BatchNorm1d(hidden_dim)
        self.__fc2 = _nn.Linear(hidden_dim, output_dim)
        self.__n2 = _nn.BatchNorm1d(output_dim)

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        """
        Forward pass of the projection head.
        :param x: The input tensor.
        :return: The output tensor.
        """
        fc1 = self.__fc1
        n1 = self.__n1
        fc2 = self.__fc2
        n2 = self.__n2

        # x shape: [B, P, C]
        B, P, C = x.shape
        x = x.view(-1, C)  # Flatten patches into batch dimension
        x = fc1(x)
        x = _F.relu(n1(x))
        x = fc2(x)
        x = n2(x)
        x = x.view(B, P, -1)  # Reshape back to [B, P, output_dim]

        return x
