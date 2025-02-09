import unittest

import torch

from src.self_supervised_learning.masked_region_loss import MaskedRegionLoss
from src.vision_transformer.model.two_scales import SemanticSegmentationVisionTransformer


class TestMaskedRegionLoss(unittest.TestCase):
    def test_forward(self):
        model = SemanticSegmentationVisionTransformer(
            image_dims=(3, 128, 128),
            num_encoder_layers=4,
            decoder_type='lightweight',
            skip_layer_ratio=4,
            encoder_dropout_rate=0.25,
            patch_fusion_dropout_rate=0.25,
            decoder_dropout_rate=0.25,
            num_encoder_heads=4,
            num_classes=1,
            patch_embedding_scale_1=(16, 1024),
            patch_embedding_scale_2=(8, 768),
        )
        x = torch.rand(2, 3, 128, 128).float()

        masked_region_loss = MaskedRegionLoss(
            model=model,
            max_patch_size=16,
            mask_ratio=0.4,
        )

        y = masked_region_loss.forward_loss(x)

        self.assertEqual(y.shape, torch.Size([]))


if __name__ == '__main__':
    unittest.main()
