import unittest

import torch

from model.config import DecoderConfig, TransformerBlockConfig, PatchEmbeddingConfig
from model.decoder import Decoder


class TestDecoder(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DecoderConfig(
            num_classes=1,
            output_dimensions=(3, 512, 512),
            max_in_channels=1024,
            transformer_block_configs=[
                TransformerBlockConfig(
                    iterations=2,
                    num_attention_heads=8,
                    window_size=(2, 2),
                    shifted_window=True,
                    dropout=False,
                ),
                TransformerBlockConfig(
                    iterations=2,
                    num_attention_heads=8,
                    window_size=(2, 2),
                    shifted_window=True,
                    dropout=False,
                ),
                TransformerBlockConfig(
                    iterations=2,
                    num_attention_heads=8,
                    window_size=(2, 2),
                    shifted_window=True,
                    dropout=False,
                ),
                TransformerBlockConfig(
                    iterations=2,
                    num_attention_heads=8,
                    window_size=(2, 2),
                    shifted_window=True,
                    dropout=False,
                ),
                TransformerBlockConfig(
                    iterations=2,
                    num_attention_heads=8,
                    window_size=(2, 2),
                    shifted_window=True,
                    dropout=False,
                ),
            ],
            patch_embedding_configs=[
                PatchEmbeddingConfig(
                    in_patches=16,
                    in_channels=1024,
                    patch_resolution=(4, 4),
                    patch_size=128,
                ),
                PatchEmbeddingConfig(
                    in_patches=64,
                    in_channels=768,
                    patch_resolution=(8, 8),
                    patch_size=64,
                ),
                PatchEmbeddingConfig(
                    in_patches=256,
                    in_channels=512,
                    patch_resolution=(16, 16),
                    patch_size=32,
                ),
                PatchEmbeddingConfig(
                    in_patches=1024,
                    in_channels=256,
                    patch_resolution=(32, 32),
                    patch_size=16,
                ),
                PatchEmbeddingConfig(
                    in_patches=4096,
                    in_channels=64,
                    patch_resolution=(64, 64),
                    patch_size=8,
                ),
            ],
        )
        self.decoder = Decoder.from_config(self.config)

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
