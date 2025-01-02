from model.attention import SwinTransformerAttention
import unittest
import torch


class TestSwinTransformerAttention(unittest.TestCase):
    def test_forward_no_shift(self) -> None:
        num_patches = 16
        vector_len = 1024

        # Test the forward pass of the Swin Transformer attention mechanism
        swin_transformer_attention = SwinTransformerAttention(
            dropout=False,
            num_heads=8,
            patch_resolution=(4, 4),
            shifted_window=False,
            vector_len=vector_len,
            window_size=(2, 2),
        )
        # - Create random patch embeddings
        patch_embeddings = torch.rand(1, num_patches, vector_len)
        # - Run the forward pass
        attended_patch_embeddings = swin_transformer_attention(patch_embeddings)
        # - Assert the output shape is correct
        self.assertEqual(attended_patch_embeddings.shape, (1, num_patches, vector_len))
        # - Assert the output is different to the input
        self.assertFalse(torch.allclose(patch_embeddings, attended_patch_embeddings))

    def test_forward_shift(self) -> None:
        num_patches = 16
        vector_len = 1024

        # Test the forward pass of the Swin Transformer attention mechanism
        swin_transformer_attention = SwinTransformerAttention(
            dropout=False,
            num_heads=8,
            patch_resolution=(4, 4),
            shifted_window=True,
            vector_len=vector_len,
            window_size=(2, 2),
        )
        # - Create random patch embeddings
        patch_embeddings = torch.rand(1, num_patches, vector_len)
        # - Run the forward pass
        attended_patch_embeddings = swin_transformer_attention(patch_embeddings)
        # - Assert the output shape is correct
        self.assertEqual(attended_patch_embeddings.shape, (1, num_patches, vector_len))
        # - Assert the output is different to the input
        self.assertFalse(torch.allclose(patch_embeddings, attended_patch_embeddings))
