import random as _random
import typing as _t

import matplotlib.pyplot as _plt
import torch as _torch
import torch.nn as _nn
import torchvision.transforms.v2 as _transforms

import src.dataset.snow as _snow
import src.self_supervised_learning.base as _ssl_base
import src.training.visualisation as _visualisation
import src.vision_transformer.model as _model


class MaskedRegionLoss(_ssl_base.SelfSupervisedLoss):
    """
    Masked Region Self-Supervised pre-training for semantic segmentation vision transformers.
    """

    def __init__(
            self,
            model: _model.SemanticSegmentationVisionTransformer,
            max_patch_size: int,
            mask_ratio: float = 0.40,
            normalise: bool = True,
    ) -> None:
        """

        :param model: The Vision Transformer Model to train.
        :param max_patch_size: The maximum size of the patches to mask.
        :param mask_ratio: The ratio of patches to mask.
        :param normalise: Whether to normalise the input image tensor.
        """
        super(MaskedRegionLoss, self).__init__(model=model)
        self.__max_patch_size = max_patch_size
        self.__mask_ratio = mask_ratio
        self.__projection_head = x = _nn.Sequential(
            _nn.Conv2d(
                model.decoder.prediction_head.in_channels, 3, kernel_size=1, stride=1
            ),
            _nn.Sigmoid(),
            _transforms.Normalize(mean=_snow.MEAN, std=_snow.STD) if normalise else _nn.Identity(),
        )
        # Initialize Weights
        self.__initialize_weights()

    def forward(self, x: _torch.Tensor) -> _t.Tuple[_torch.Tensor, _torch.Tensor]:
        """
        Forward Pass for the Masked Region Loss pre-training. This method masks random patches of a given size in the
        input tensor and then applies the model's encoder and decoder to the masked tensor to reconstruct the original
        image.

        :param x: The input tensor.
        :return: The reconstructed image and the mask.
        """
        model = self.model
        projection_head = self.__projection_head

        # Mask Image
        masked_image, mask = self._mask_image(x)

        # Forward Pass
        # - Patch Embedding
        patch_embeddings = model.apply_patch_embedding_stage(masked_image)
        # - Encoder Stage
        patch_embeddings = model.apply_encoder_stage(patch_embeddings=patch_embeddings)
        # - Decoder Stage
        reconstructed = model.decoder.forward(patch_embeddings, apply_prediction_head=False)
        # - Apply temporary projection head to RGB
        reconstructed = projection_head(reconstructed)

        assert reconstructed.shape == x.shape, f"Expected {x.shape} but got {reconstructed.shape}"

        return reconstructed, mask

    def forward_loss(self, x: _torch.Tensor) -> _torch.Tensor:
        """
        Forward Pass for the Masked Region Loss pre-training. This method wraps the forward pass and calculates the
        loss between the reconstructed image and the original image.

        :param x: The input tensor.
        :return: The loss.
        """
        reconstructed, mask = self.forward(x)
        mask = 1 - mask
        loss = self.loss_fn(reconstructed, x, mask)

        return loss

    def _mask_image(self, x: _torch.Tensor) -> _t.Tuple[_torch.Tensor, _torch.Tensor]:
        """
        Randomly mask patches of the input image tensor.

        :param x: The input image tensor.
        :return: The masked image and the mask.
        """
        max_patch_size = self.__max_patch_size
        mask_ratio = self.__mask_ratio

        B, C, H, W = x.shape
        masked_tensor = x.clone()

        # Calculate the number of patches along each dimension
        num_patches_h = H // max_patch_size
        num_patches_w = W // max_patch_size
        total_patches = num_patches_h * num_patches_w

        # Determine how many patches to mask
        num_patches_to_mask = int(total_patches * mask_ratio)

        # Create a binary mask initialized to all ones
        mask = _torch.ones((1, 1, H, W), device=x.device)

        # Randomly select patches to mask
        patches = [(i, j) for i in range(num_patches_h) for j in range(num_patches_w)]
        random_patches = _random.sample(patches, num_patches_to_mask)

        # Mask selected patches
        for i, j in random_patches:
            h_start = i * max_patch_size
            h_end = h_start + max_patch_size
            w_start = j * max_patch_size
            w_end = w_start + max_patch_size
            mask[:, :, h_start:h_end, w_start:w_end] = 0

        masked_tensor = masked_tensor * mask

        return masked_tensor, mask

    def loss_fn(self, reconstructed: _torch.Tensor, original: _torch.Tensor, mask: _torch.Tensor) -> _torch.Tensor:
        """
        Calculate the MSE loss between the reconstructed and original image.

        :param reconstructed: The reconstructed image.
        :param original: The original image.
        :return: The loss.
        """
        mask_ratio = self.__mask_ratio

        loss = _torch.mean((reconstructed - original) ** 2 * mask) / mask_ratio

        return loss

    def __initialize_weights(self) -> None:
        """
        Initialize the weights of the masked region loss module.
        """
        projection_head = self.__projection_head

        for layer in projection_head:
            if isinstance(layer, _nn.ConvTranspose2d):
                _nn.init.kaiming_normal_(layer.weight)
                _nn.init.zeros_(layer.bias)


def visualise_masked_region_prediction(model: MaskedRegionLoss, image: _torch.Tensor, normalise: bool = True) -> None:
    """
    Visualise the Masked Region Prediction for a given image.

    :param model: The Masked Region Loss model.
    :param image: The image tensor.
    :param normalise: Whether to normalise the image tensor
    """
    model.eval()
    with _torch.no_grad():
        image_ = _transforms.Normalize(mean=_snow.MEAN, std=_snow.STD)(image) if normalise else image
        reconstructed_image, mask = model.forward(image_.unsqueeze(0))
        reconstructed_image = reconstructed_image.squeeze(0)

    # Denormalize the reconstructed_image for visualisation
    if normalise:
        dtype = image.dtype
        device = image.device
        mean = _torch.tensor(_snow.MEAN, dtype=dtype, device=device).view(-1, 1, 1)
        std = _torch.tensor(_snow.STD, dtype=dtype, device=device).view(-1, 1, 1)
        reconstructed_image_ = (reconstructed_image * std) + mean
    else:
        reconstructed_image_ = reconstructed_image

    fig, axes = _plt.subplots(2, 6, figsize=(20, 10))  # Adjusted to fit all images
    axes = axes.flatten()  # Convert to a 1D list for easier iteration

    mask = mask.squeeze(0)
    for ax, (title, tensor) in zip(
            axes,
            (
                    # Input Image
                    ("Input Image", image),
                    ("Input Image Normalised", image_),

                    ("Input Patches Image", image * mask),
                    ("Input Patches Image Normalised", image_ * mask),

                    ("Missing Patches", image * (1 - mask)),
                    ("Missing Patches Normalised", image_ * (1 - mask)),

                    # Reconstructed Image
                    ("Reconstructed Image", reconstructed_image_),
                    ("Reconstructed Image Normalised", reconstructed_image),

                    ("Reconstructed Input Patches", reconstructed_image_ * mask),
                    ("Reconstructed Input Patches Normalised", reconstructed_image * mask),

                    ("Reconstructed Missing Patches", reconstructed_image_ * (1 - mask)),
                    ("Reconstructed Missing Patches Normalised", reconstructed_image * (1 - mask)),
            )
    ):
        ax.set_title(title, fontsize=10)  # Use correct title
        _visualisation.display_tensor_image(tensor, ax=ax)  # Use correct tensor

    _plt.tight_layout()
    _plt.show()
