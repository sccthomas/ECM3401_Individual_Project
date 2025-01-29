import unittest

import torch

from src.vision_transformer.common.patch_fusion import PatchFusion


class TestPatchFusion(unittest.TestCase):
    def test_forward(self) -> None:
        patch_fusion = PatchFusion(
            in_embed=768,
            out_embed=512,
        )

        x1 = torch.randn(10, 256, 768)
        x2 = torch.randn(10, 1024, 512)

        y = patch_fusion(x1, x2)

        self.assertEqual(y.shape, (10, 1024, 512))

        patch_fusion = PatchFusion(
            in_embed=512,
            out_embed=768,
        )

        x1 = torch.randn(10, 1024, 512)
        x2 = torch.randn(10, 256, 768)

        y = patch_fusion(x1, x2)

        self.assertEqual(y.shape, (10, 256, 768))


if __name__ == '__main__':
    unittest.main()
