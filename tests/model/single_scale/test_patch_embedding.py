import unittest

import torch

from src.model.single_scale.patch_embedding import PatchEmbedding


class TestPatchEmbedding(unittest.TestCase):
    def test_forward(self) -> None:
        device = torch.device('mps')
        image = torch.randn(10, 3, 256, 256).to(device)
        B, C, H, _ = image.shape

        patch_embedding = PatchEmbedding(in_channels=C, embed_dim=768, patch_size=16, image_size=H).to(device)
        patch_embedding = patch_embedding(image)
        self.assertEqual(patch_embedding.shape, torch.Size([B, 256, 768]))
        self.assertTrue(patch_embedding.is_contiguous())


if __name__ == '__main__':
    unittest.main()
