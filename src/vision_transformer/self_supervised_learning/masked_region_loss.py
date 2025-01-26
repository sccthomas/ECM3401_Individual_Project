import random as _random
import typing as _t

import torch as _torch
import torch.nn as _nn

import src.vision_transformer.model.base as _base
import src.vision_transformer.self_supervised_learning.base as _ssl_base


class MaskedRegionLoss(_ssl_base.SelfSupervisedLoss):
    """
    Masked Region Self-Supervised pre-training for semantic segmentation vision transformers.
    """

    def __init__(
            self,
            model: _base.SemanticSegmentationVisionTransformerBase,
            max_patch_size: int,
            mask_ratio: float = 0.40,
    ) -> None:
        """

        :param model: The Vision Transformer Model to train.
        :param max_patch_size: The maximum size of the patches to mask.
        :param mask_ratio: The ratio of patches to mask.
        """
        super(MaskedRegionLoss, self).__init__(model=model)
        self.__max_patch_size = max_patch_size
        self.__mask_ratio = mask_ratio
        self.__projection_head = x = _nn.Sequential(
            _nn.Conv2d(
                model.decoder.prediction_head.in_channels, 3, kernel_size=1, stride=1
            ),
            _nn.Sigmoid(),
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
        kwargs = model.apply_patch_embedding_stage(x)
        # - Encoder Stage
        kwargs = model.apply_encoder_stage(**kwargs)
        # - Decoder Stage
        reconstructed = model.apply_decoder_fusion(**kwargs)
        reconstructed = model.decoder.apply_transposed_convolutions(reconstructed)
        reconstructed = projection_head(reconstructed)

        assert x.shape == reconstructed.shape, f"Expected {x.shape} but got {reconstructed.shape}"

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
        loss = self.__loss_fn(reconstructed, x, mask)

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

    def __loss_fn(self, reconstructed: _torch.Tensor, original: _torch.Tensor, mask: _torch.Tensor) -> _torch.Tensor:
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
