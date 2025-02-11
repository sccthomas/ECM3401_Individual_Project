import unittest

import torch

from src.vision_transformer.common.patch_fusion import PatchFusion


class TestPatchFusion(unittest.TestCase):
    def test_forward(self) -> None:
        patch_fusion = PatchFusion(
            in_dims=[[1024, 768], [4096, 512]],
            out_patches=256,
            out_embed=1024,
            dropout_rate=0.25,
        )

        x1 = torch.randn(10, 256, 1024)
        x2 = torch.randn(10, 1024, 768)
        x3 = torch.randn(10, 4096, 512)

        y = patch_fusion(target_tensor=x1, tensors=[x2, x3])

        self.assertEqual(y.shape, x1.shape)

    def test_forward_gated_attention(self) -> None:
        patch_fusion = PatchFusion(
            in_dims=[[1024, 768], [4096, 512]],
            out_patches=256,
            out_embed=1024,
            dropout_rate=0.25,
            use_gated_attention=True,
            num_heads=2,
        )

        x1 = torch.randn(10, 256, 1024)
        x2 = torch.randn(10, 1024, 768)
        x3 = torch.randn(10, 4096, 512)

        y = patch_fusion(target_tensor=x1, tensors=[x2, x3])

        self.assertEqual(y.shape, x1.shape)


if __name__ == '__main__':
    unittest.main()
