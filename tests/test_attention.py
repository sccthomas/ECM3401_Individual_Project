from model.attention import SwinTransformerAttention
import unittest
import torch


class TestSwinTransformerAttention(unittest.TestCase):
    def test_forward_no_shift(self) -> None:
        in_patches = 16
        in_channels = 1024

        # Test the forward pass of the Swin Transformer attention mechanism
        swin_transformer_attention = SwinTransformerAttention(
            in_patches=in_patches,
            in_channels=in_channels,
            dropout=False,
            num_attention_heads=8,
            patch_resolution=(4, 4),
            shifted_window=False,
            window_size=(2, 2),
        )
        # - Create random patch embeddings
        patch_embeddings = torch.rand(1, in_patches, in_channels)
        # - Run the forward pass
        attended_patch_embeddings = swin_transformer_attention(patch_embeddings)
        # - Assert the output shape is correct
        self.assertEqual(attended_patch_embeddings.shape, (1, in_patches, in_channels))
        # - Assert the output is different to the input
        self.assertFalse(torch.allclose(patch_embeddings, attended_patch_embeddings))

    def test_forward_shift(self) -> None:
        in_patches = 16
        in_channels = 1024

        # Test the forward pass of the Swin Transformer attention mechanism
        swin_transformer_attention = SwinTransformerAttention(
            in_patches=in_patches,
            in_channels=in_channels,
            dropout=False,
            num_attention_heads=8,
            patch_resolution=(4, 4),
            shifted_window=True,
            window_size=(2, 2),
        )
        # - Create random patch embeddings
        patch_embeddings = torch.rand(1, in_patches, in_channels)
        # - Run the forward pass
        attended_patch_embeddings = swin_transformer_attention(patch_embeddings)
        # - Assert the output shape is correct
        self.assertEqual(attended_patch_embeddings.shape, (1, in_patches, in_channels))
        # - Assert the output is different to the input
        self.assertFalse(torch.allclose(patch_embeddings, attended_patch_embeddings))
