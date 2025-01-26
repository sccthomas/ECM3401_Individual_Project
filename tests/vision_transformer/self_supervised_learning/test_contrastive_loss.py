import unittest

import torch

from src.vision_transformer.model.two_scales import SemanticSegmentationVisionTransformer
from src.vision_transformer.self_supervised_learning.contrastive_loss import ContrastivePreTraining


class TestContrastivePreTraining(unittest.TestCase):
    def test_forward(self) -> None:
        model = SemanticSegmentationVisionTransformer(
            image_dims=(3, 256, 256),
            num_encoder_layers=12,
            patch_embedding_scale_1=(16, 1024),
            patch_embedding_scale_2=(8, 768),
        )
        x = torch.rand(2, 3, 256, 256).float()

        contrastive_model = ContrastivePreTraining(
            model=model,
            encoder_dims=[1024, 768],
            projection_dim=128,
        )

        y = contrastive_model(x)

        self.assertEqual(y.shape, torch.Size([]))


if __name__ == '__main__':
    unittest.main()
