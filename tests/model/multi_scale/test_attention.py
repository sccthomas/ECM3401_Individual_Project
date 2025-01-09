import unittest

import torch
import torch.nn as nn
import torch.optim as optim

from src.model.multi_scale.attention import SwinTransformerAttention


class TestSwinTransformerAttention(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device('mps')
        self.in_patches = 16
        self.in_channels = 1024

    def test_forward_no_shift(self) -> None:
        device = self.device
        in_patches = self.in_patches
        in_channels = self.in_channels

        # Define the Swin Transformer attention mechanism
        swin_transformer_attention = SwinTransformerAttention(
            in_patches=in_patches,
            in_channels=in_channels,
            dropout=True,
            num_attention_heads=8,
            patch_resolution=(4, 4),
            shifted_window=False,
            window_size=(2, 2),
        ).to(device)

        # Create random patch embeddings
        patch_embeddings = torch.rand(10, in_patches, in_channels).to(device)
        target = torch.rand(10, in_patches, in_channels).to(device)

        # Define a loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(swin_transformer_attention.parameters(), lr=0.001)

        # Training step
        optimizer.zero_grad()
        attended_patch_embeddings = swin_transformer_attention(patch_embeddings)
        loss = criterion(attended_patch_embeddings, target)
        loss.backward()
        optimizer.step()

        # Assertions
        self.assertEqual(attended_patch_embeddings.shape, (10, in_patches, in_channels))
        self.assertFalse(torch.allclose(patch_embeddings, attended_patch_embeddings))
        self.assertGreater(loss.item(), 0)

    def test_forward_shift(self) -> None:
        device = self.device
        in_patches = self.in_patches
        in_channels = self.in_channels

        # Define the Swin Transformer attention mechanism
        swin_transformer_attention = SwinTransformerAttention(
            in_patches=in_patches,
            in_channels=in_channels,
            dropout=True,
            num_attention_heads=8,
            patch_resolution=(4, 4),
            shifted_window=True,
            window_size=(2, 2),
        ).to(device)

        # Create random patch embeddings
        patch_embeddings = torch.rand(10, in_patches, in_channels).to(device)
        target = torch.rand(10, in_patches, in_channels).to(device)

        # Define a loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(swin_transformer_attention.parameters(), lr=0.001)

        # Training step
        optimizer.zero_grad()
        attended_patch_embeddings = swin_transformer_attention(patch_embeddings)
        loss = criterion(attended_patch_embeddings, target)
        loss.backward()
        optimizer.step()

        # Assertions
        self.assertEqual(attended_patch_embeddings.shape, (10, in_patches, in_channels))
        self.assertFalse(torch.allclose(patch_embeddings, attended_patch_embeddings))
        self.assertGreater(loss.item(), 0)


if __name__ == '__main__':
    unittest.main()
