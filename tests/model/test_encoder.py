import unittest

import torch

from src.model.config import EncoderConfig, TransformerBlockConfig, PatchEmbeddingConfig
from src.model.encoder import Encoder


class TestEncoder(unittest.TestCase):
    def setUp(self):
        self.config = EncoderConfig(
            num_stages=2,
            transformer_block_configs=[
                [
                    TransformerBlockConfig(
                        iterations=3,
                        num_attention_heads=8,
                        window_size=(2, 2),
                        shifted_window=True,
                        dropout=True,
                    )
                    for _ in range(2)
                ],
                [
                    TransformerBlockConfig(
                        iterations=3,
                        num_attention_heads=8,
                        window_size=(4, 4),
                        shifted_window=True,
                        dropout=True,
                    )
                    for _ in range(2)
                ],
                [
                    TransformerBlockConfig(
                        iterations=3,
                        num_attention_heads=8,
                        window_size=(4, 4),
                        shifted_window=True,
                        dropout=True,
                    )
                    for _ in range(2)
                ],
                [
                    TransformerBlockConfig(
                        iterations=3,
                        num_attention_heads=8,
                        window_size=(8, 8),
                        shifted_window=True,
                        dropout=True,
                    )
                    for _ in range(2)
                ],
                [
                    TransformerBlockConfig(
                        iterations=3,
                        num_attention_heads=8,
                        window_size=(16, 16),
                        shifted_window=True,
                        dropout=True,
                    )
                    for _ in range(2)
                ],
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
            ]
        )
        self.device = torch.device('mps')
        self.encoder = Encoder.from_config(self.config).to(self.device)

    def test_transformer_blocks(self) -> None:
        device = self.device
        encoder = self.encoder
        transformer_blocks = encoder.transformer_blocks
        patch_embedding_configs = self.config.patch_embedding_configs

        self.assertEqual(len(transformer_blocks), 2)
        for i in range(2):
            self.assertEqual(len(transformer_blocks[i]), 5)
            for j in range(5):
                self.assertEqual(transformer_blocks[i][j].in_channels, patch_embedding_configs[j].in_channels)
                self.assertEqual(transformer_blocks[i][j].in_patches, patch_embedding_configs[j].in_patches)

    def test_skip_connections(self) -> None:
        encoder = self.encoder
        skip_connections = encoder.skip_connections

        self.assertEqual(len(skip_connections), 2)

        expected_skip_connections_scales = [
            [
                (768, 1024),
                (512, 1024),
                (256, 1024),
                (64, 1024),
            ], [
                (1024, 768),
                (512, 768),
                (256, 768),
                (64, 768),
            ], [
                (1024, 512),
                (768, 512),
                (256, 512),
                (64, 512),
            ], [
                (1024, 256),
                (768, 256),
                (512, 256),
                (64, 256),
            ], [
                (1024, 64),
                (768, 64),
                (512, 64),
                (256, 64),
            ]
        ]

        for i in range(2):
            self.assertEqual(len(skip_connections[i]), 5)
            for j in range(5):
                actual_skip_connection_scales = [
                    (linear_operation.in_features, linear_operation.out_features)
                    for linear_operation in skip_connections[i][j].linear_operations
                ]
                self.assertEqual(actual_skip_connection_scales, expected_skip_connections_scales[j])

    def test_forward(self) -> None:
        device = self.device
        config = self.config
        encoder = self.encoder
        patch_embedding_configs = config.patch_embedding_configs

        # Test forward pass
        batch_size = 5
        patch_embeddings = [
            torch.rand(batch_size, patch_embedding_config.in_patches, patch_embedding_config.in_channels).to(device)
            for patch_embedding_config in patch_embedding_configs
        ]

        output = encoder(patch_embeddings)

        self.assertEqual(len(output), 5)
        for i in range(5):
            self.assertEqual(output[i].shape, patch_embeddings[i].shape)


if __name__ == '__main__':
    unittest.main()
