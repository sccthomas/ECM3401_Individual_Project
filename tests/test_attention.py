import unittest

import torch

from model.attention import SwinTransformerAttention


class TestSwinTransformerAttention(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device('mps')
        self.in_patches = 16
        self.in_channels = 1024

    def test_forward_no_shift(self) -> None:
        device = self.device
        in_patches = self.in_patches
        in_channels = self.in_channels

        # Test the forward pass of the Swin Transformer attention mechanism
        swin_transformer_attention = SwinTransformerAttention(
            in_patches=in_patches,
            in_channels=in_channels,
            dropout=True,
            num_attention_heads=8,
            patch_resolution=(4, 4),
            shifted_window=False,
            window_size=(2, 2),
        )
        swin_transformer_attention = swin_transformer_attention.to(device)
        # - Create random patch embeddings
        patch_embeddings = torch.rand(1, in_patches, in_channels)
        patch_embeddings = patch_embeddings.to(device)
        # - Run the forward pass
        attended_patch_embeddings = swin_transformer_attention(patch_embeddings)
        # - Assert the output shape is correct
        self.assertEqual(attended_patch_embeddings.shape, (1, in_patches, in_channels))
        # - Assert the output is different to the input
        self.assertFalse(torch.allclose(patch_embeddings, attended_patch_embeddings))

    def test_forward_shift(self) -> None:
        device = self.device
        in_patches = self.in_patches
        in_channels = self.in_channels

        # Test the forward pass of the Swin Transformer attention mechanism
        swin_transformer_attention = SwinTransformerAttention(
            in_patches=in_patches,
            in_channels=in_channels,
            dropout=True,
            num_attention_heads=8,
            patch_resolution=(4, 4),
            shifted_window=True,
            window_size=(2, 2),
        )
        swin_transformer_attention = swin_transformer_attention.to(device)
        # - Create random patch embeddings
        patch_embeddings = torch.rand(1, in_patches, in_channels)
        patch_embeddings = patch_embeddings.to(device)
        # - Run the forward pass
        attended_patch_embeddings = swin_transformer_attention(patch_embeddings)
        # - Assert the output shape is correct
        self.assertEqual(attended_patch_embeddings.shape, (1, in_patches, in_channels))
        # - Assert the output is different to the input
        self.assertFalse(torch.allclose(patch_embeddings, attended_patch_embeddings))
