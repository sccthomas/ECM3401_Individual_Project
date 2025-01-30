import unittest

import torch

from src.vision_transformer.model.three_scales import SemanticSegmentationVisionTransformer


class TestSemanticSegmentationVisionTransformer(unittest.TestCase):
    def test_forward(self) -> None:
        model = SemanticSegmentationVisionTransformer(
            image_dims=(3, 256, 256),
            num_encoder_layers=12,
            patch_embedding_scale_1=(32, 1024),
            patch_embedding_scale_2=(16, 768),
            patch_embedding_scale_3=(8, 512),
        )

        x = torch.rand(16, 3, 256, 256).float()

        y = model(x)

        self.assertEqual(y.shape, (16, 1, 256, 256))
        self.assertFalse(torch.isnan(y).any())


if __name__ == '__main__':
    unittest.main()
