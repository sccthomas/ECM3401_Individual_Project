import unittest

import torch
from torchvision.transforms.v2 import Normalize

from src.vision_transformer.model.three_scales import SemanticSegmentationVisionTransformer


class TestSemanticSegmentationVisionTransformer(unittest.TestCase):
    def test_forward(self) -> None:
        model = SemanticSegmentationVisionTransformer(
            image_dims=(3, 256, 256),
            patch_embedding_scale_1=(32, 1024),
            patch_embedding_scale_2=(16, 768),
            patch_embedding_scale_3=(8, 512),
        )

        x = torch.randint(0, 255, (16, 3, 256, 256)).float() / 255
        norm = Normalize(mean=[0.4808, 0.4178, 0.5046], std=[0.2767, 0.2698, 0.2856])
        x = norm(x)

        y = model(x)

        self.assertEqual(y.shape, (16, 1, 256, 256))
        self.assertFalse(torch.isnan(y).any())


if __name__ == '__main__':
    unittest.main()
