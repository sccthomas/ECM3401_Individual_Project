import unittest

import torch

from src.vision_transformer.model.two_scales import SemanticSegmentationVisionTransformer
from src.vision_transformer.self_supervised_learning.masked_region_loss import MaskedRegionLoss


class TestMaskedRegionLoss(unittest.TestCase):
    def test_forward(self):
        model = SemanticSegmentationVisionTransformer(
            image_dims=(3, 256, 256),
            num_encoder_layers=12,
            patch_embedding_scale_1=(16, 1024),
            patch_embedding_scale_2=(8, 768),
        )
        x = torch.rand(2, 3, 256, 256).float()

        masked_region_loss = MaskedRegionLoss(
            model=model,
            max_patch_size=16,
            mask_ratio=0.4,
        )

        y = masked_region_loss(x)


if __name__ == '__main__':
    unittest.main()
