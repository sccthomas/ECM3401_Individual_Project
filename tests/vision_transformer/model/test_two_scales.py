import unittest

import torch

from src.vision_transformer.model.two_scales import SemanticSegmentationVisionTransformer


class TestSemanticSegmentationVisionTransformer(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("mps" if torch.cuda.is_available() else "cpu")
        self.model = SemanticSegmentationVisionTransformer(
            image_dims=(3, 256, 256),
            num_encoder_layers=12,
            patch_embedding_scale_1=(16, 1024),
            patch_embedding_scale_2=(8, 768),
        ).to(self.device)

    def test_forward(self) -> None:
        model = self.model
        device = self.device

        x = torch.rand(16, 3, 256, 256).float().to(device)

        y = model(x)

        self.assertEqual(y.shape, (16, 1, 256, 256))
        self.assertFalse(torch.isnan(y).any())

    def test_forward_with_weights(self) -> None:
        model = self.model
        device = self.device

        x = torch.rand(16, 3, 256, 256).float().to(device)

        y, weights = model(x, return_attention_weights=True)

        self.assertEqual(y.shape, (16, 1, 256, 256))
        self.assertFalse(torch.isnan(y).any())

        self.assertEqual(len(weights), 2)
        self.assertEqual(len(weights["x1"]), 12)
        self.assertEqual(len(weights["x2"]), 12)
        self.assertIsNotNone(weights["x1"][0])


if __name__ == '__main__':
    unittest.main()
