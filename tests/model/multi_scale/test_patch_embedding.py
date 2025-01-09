import unittest

import torch

from src.model.multi_scale.patch_embedding import PatchEmbedding


class MyTestCase(unittest.TestCase):
    def test_forward(self) -> None:
        device = torch.device('mps')
        image = torch.randn(10, 3, 512, 512).to(device)
        B, C, H, _ = image.shape

        patch_embedding = PatchEmbedding(in_channels=C, out_channels=1024, patch_size=128, image_size=H).to(device)
        patch_embedding = patch_embedding(image)
        self.assertEqual(patch_embedding.shape, torch.Size([B, 16, 1024]))
        self.assertTrue(patch_embedding.is_contiguous())

        patch_embedding = PatchEmbedding(in_channels=C, out_channels=768, patch_size=64, image_size=H).to(device)
        patch_embedding = patch_embedding(image)
        self.assertEqual(patch_embedding.shape, torch.Size([B, 64, 768]))
        self.assertTrue(patch_embedding.is_contiguous())

        patch_embedding = PatchEmbedding(in_channels=C, out_channels=512, patch_size=32, image_size=H).to(device)
        patch_embedding = patch_embedding(image)
        self.assertEqual(patch_embedding.shape, torch.Size([B, 256, 512]))
        self.assertTrue(patch_embedding.is_contiguous())

        patch_embedding = PatchEmbedding(in_channels=C, out_channels=256, patch_size=16, image_size=H).to(device)
        patch_embedding = patch_embedding(image)
        self.assertEqual(patch_embedding.shape, torch.Size([B, 1024, 256]))
        self.assertTrue(patch_embedding.is_contiguous())

        patch_embedding = PatchEmbedding(in_channels=C, out_channels=64, patch_size=8, image_size=H).to(device)
        patch_embedding = patch_embedding(image)
        self.assertEqual(patch_embedding.shape, torch.Size([B, 4096, 64]))
        self.assertTrue(patch_embedding.is_contiguous())


if __name__ == '__main__':
    unittest.main()
