import unittest

import torch

from src.self_supervised_learning.contrastive_loss import ContrastivePreTraining
from src.vision_transformer.model.two_scales import SemanticSegmentationVisionTransformer


class TestContrastivePreTraining(unittest.TestCase):
    def test_forward(self) -> None:
        model = SemanticSegmentationVisionTransformer(
            image_dims=(3, 128, 128),
            num_encoder_layers=4,
            encoder_dropout_rate=0.25,
            patch_fusion_dropout_rate=0.25,
            decoder_dropout_rate=0.25,
            num_encoder_heads=4,
            num_classes=1,
            patch_embedding_scale_1=(16, 1024),
            patch_embedding_scale_2=(8, 768),
        )
        x = torch.rand(2, 3, 128, 128).float()

        contrastive_model = ContrastivePreTraining(
            model=model,
            encoder_dims=[1024, 768],
            projection_dim=128,
        )

        y = contrastive_model.forward_loss(x)

        self.assertEqual(y.shape, torch.Size([]))


if __name__ == '__main__':
    unittest.main()
