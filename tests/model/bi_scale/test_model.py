import unittest

import torch

from src.model.bi_scale.model import SemanticSegmentationVisionTransformer


class TestSemanticSegmentationVisionTransformer(unittest.TestCase):
    def test_forward(self) -> None:
        model = SemanticSegmentationVisionTransformer(
            image_dims=(3, 256, 256),
            patch_embedding_scale_1=(16, 1024),
            patch_embedding_scale_2=(8, 512),
        )

        x = torch.rand(10, 3, 256, 256)

        y = model(x)

        self.assertEqual(y.shape, (10, 1, 256, 256))


if __name__ == '__main__':
    unittest.main()
