import unittest

import torch

from src.model.single_scale.model import SemanticSegmentationVisionTransformer


class TestSemanticSegmentationVisionTransformer(unittest.TestCase):
    def test_forward(self) -> None:
        device = torch.device('mps')
        model = SemanticSegmentationVisionTransformer(
            in_channels=3,
            num_classes=1,
            embed_dim=768,
            patch_size=16,
            img_size=256,
            num_heads=12,
            num_layers=12
        ).to(device)

        input_tensor = torch.randn(8, 3, 256, 256).to(device)  # Batch of 8 images
        output = model(input_tensor)

        self.assertEqual(output.shape, (8, 1, 256, 256))


if __name__ == '__main__':
    unittest.main()
