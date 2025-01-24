import torch as _torch

import src.vision_transformer.model.base as _base
import src.vision_transformer.self_supervised_learning.base as _ssl_base


# Could we get away with only using one learnable operation in patch fusion so that the number of patches dimension
# is not statically defined?
# Potentially we could move the F.Interpolate with nearest to be in the forward method and be entirely dynamic since we
# both tensor shapes. Therefore, there is no need for the number of patches kwarg in the constructor.
# 1. We can split the patch embeddings into the 75% and 25% split
# 2. We can then encode the patch embeddings since there is no deependence on the number of patches
# 3. Then add the patches back together.
# 4. Then pass them into the decoder layer with the correct number of patches (total). 

class MaskedRegionLoss(_ssl_base.SelfSupervisedLoss):
    def __init__(
            self,
            model: _base.SemanticSegmentationVisionTransformerBase,
            mask_ratio=0.75,
    ) -> None:
        super(MaskedRegionLoss, self).__init__(model=model)

        self.__mask_ratio = mask_ratio

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        model = self.model

        # - Mask image
        masked_image = self.__mask_image(x)

        # - Forward pass
        predicted_mask = model(masked_image)

        # - Calculate loss
        loss = self.__loss_fn(reconstucted_image, x)

        return loss

    def __mask_image(self, x: _torch.Tensor) -> _torch.Tensor:
        mask_ratio = self.__mask_ratio

        # Compute the number of pixels to mask
        _, H, W = x.shape
        num_mask_pixels = int((mask_ratio / 100) * H * W)

        # Generate a random mask
        mask = _torch.rand(H, W) < (num_mask_pixels / (H * W))

        # Apply the mask to the image (broadcast across channels)
        masked_image = x.clone()
        masked_image[:, mask] = 0  # Replace masked pixels with 0 (black)

        return masked_image

    def __loss_fn(self, reconstructed: _torch.Tensor, original: _torch.Tensor) -> _torch.Tensor:
        """

        :param reconstructed:
        :param original:
        :return:
        """
