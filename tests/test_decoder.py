import unittest
from model.decoder import Decoder
import torch
from model.transformer import TransformerBlockDecoder
import torch.nn as nn


class TestDecoder(unittest.TestCase):
    def setUp(self) -> None:
        config = {

        }
        self.decoder = Decoder(
            max_in_channels=1024,
            num_classes=1,
            output_dimensions=(512, 512),
            transformer_blocks=nn.ModuleList([
                TransformerBlockDecoder(
                    in_patches=4096,
                    in_channels=64,
                    patch_resolution=(64, 64),
                    output_dims=(1024, 256),
                    iterations=2,
                    num_attention_heads=8,
                    window_size=(16, 16),
                    shifted_window=True,
                    dropout=False,
                ),
                TransformerBlockDecoder(
                    in_patches=1024,
                    in_channels=256,
                    patch_resolution=(32, 32),
                    output_dims=(256, 512),
                    iterations=2,
                    num_attention_heads=8,
                    window_size=(8, 8),
                    shifted_window=True,
                    dropout=False,
                ),
                TransformerBlockDecoder(
                    in_patches=256,
                    in_channels=512,
                    patch_resolution=(16, 16),
                    output_dims=(64, 768),
                    iterations=2,
                    num_attention_heads=8,
                    window_size=(4, 4),
                    shifted_window=True,
                    dropout=False,
                ),
                TransformerBlockDecoder(
                    in_patches=64,
                    in_channels=768,
                    patch_resolution=(8, 8),
                    output_dims=(16, 1024),
                    iterations=2,
                    num_attention_heads=8,
                    window_size=(4, 4),
                    shifted_window=True,
                    dropout=False,
                ),
                TransformerBlockDecoder(
                    in_patches=16,
                    in_channels=1024,
                    patch_resolution=(4, 4),
                    output_dims=(16, 1024),
                    iterations=2,
                    num_attention_heads=8,
                    window_size=(2, 2),
                    shifted_window=True,
                    dropout=False,
                ),
            ])
        )

    def test_forward(self) -> None:
        decoder = self.decoder

        batch_size = 2
        patch_embeddings = [
            torch.randn(batch_size, 16, 1024),
            torch.randn(batch_size, 64, 768),
            torch.randn(batch_size, 256, 512),
            torch.randn(batch_size, 1024, 256),
            torch.randn(batch_size, 4096, 64),
        ]

        output = decoder(patch_embeddings)

        self.assertEqual(output.shape, (batch_size, 1, 512, 512))


if __name__ == '__main__':
    unittest.main()
