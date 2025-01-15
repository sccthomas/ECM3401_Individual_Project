import unittest

import torch
import torch.nn as nn
import torch.optim as optim

from src.vision_transformer.model.multi_scale.config import DecoderConfig, TransformerBlockConfig, PatchEmbeddingConfig
from src.vision_transformer.model.multi_scale.decoder import Decoder


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
                    dropout=True,
                ),
                TransformerBlockConfig(
                    iterations=2,
                    num_attention_heads=8,
                    window_size=(4, 4),
                    shifted_window=True,
                    dropout=True,
                ),
                TransformerBlockConfig(
                    iterations=2,
                    num_attention_heads=8,
                    window_size=(4, 4),
                    shifted_window=True,
                    dropout=True,
                ),
                TransformerBlockConfig(
                    iterations=2,
                    num_attention_heads=8,
                    window_size=(8, 8),
                    shifted_window=True,
                    dropout=True,
                ),
                TransformerBlockConfig(
                    iterations=2,
                    num_attention_heads=8,
                    window_size=(16, 16),
                    shifted_window=True,
                    dropout=True,
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
        self.device = torch.device('mps')
        self.decoder = Decoder.from_config(self.config).to(self.device)

    def test_forward(self) -> None:
        device = self.device
        decoder = self.decoder

        batch_size = 5
        patch_embeddings = [
            torch.randn(batch_size, 16, 1024).to(device),
            torch.randn(batch_size, 64, 768).to(device),
            torch.randn(batch_size, 256, 512).to(device),
            torch.randn(batch_size, 1024, 256).to(device),
            torch.randn(batch_size, 4096, 64).to(device),
        ]
        target = torch.randint(0, 2, (batch_size, 1, 512, 512)).float().to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(decoder.parameters(), lr=0.001)
        output = decoder(patch_embeddings)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        self.assertEqual(output.shape, (batch_size, 1, 512, 512))


if __name__ == '__main__':
    unittest.main()
