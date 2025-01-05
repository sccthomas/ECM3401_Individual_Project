from model.encoder import Encoder
from model.config import EncoderConfig
import unittest
import torch


class TestEncoder(unittest.TestCase):
    def setUp(self):
        self.config = EncoderConfig(
            num_stages=2,
            input_dimensions=(3, 512, 512),
            patch_embedding_configs=[
                {
                    "patch_embedding_info": {
                        'patch_size': 128,
                        'in_channels': 1024,
                    },
                    "transformer_block_configs": [
                        {
                            'dropout': False,
                            'iterations': 3,
                            'num_attention_heads': 8,
                            'shifted_window': False,
                            'window_size': (2, 2)
                        },
                        {
                            'dropout': True,
                            'iterations': 3,
                            'num_attention_heads': 8,
                            'shifted_window': False,
                            'window_size': (2, 2)
                        },
                    ],
                },
                {
                    "patch_embedding_info": {
                        'patch_size': 64,
                        'in_channels': 768,
                    },
                    "transformer_block_configs": [
                        {
                            'dropout': True,
                            'iterations': 3,
                            'num_attention_heads': 8,
                            'shifted_window': False,
                            'window_size': (2, 2)
                        },
                        {
                            'dropout': True,
                            'iterations': 3,
                            'num_attention_heads': 8,
                            'shifted_window': False,
                            'window_size': (2, 2)
                        },
                    ],
                },
                {
                    "patch_embedding_info": {
                        'patch_size': 32,
                        'in_channels': 512,
                    },
                    "transformer_block_configs": [
                        {
                            'dropout': True,
                            'iterations': 3,
                            'num_attention_heads': 8,
                            'shifted_window': False,
                            'window_size': (2, 2)
                        },
                        {
                            'dropout': True,
                            'iterations': 3,
                            'num_attention_heads': 8,
                            'shifted_window': False,
                            'window_size': (2, 2)
                        },
                    ],
                },
                {
                    "patch_embedding_info": {
                        'patch_size': 16,
                        'in_channels': 256,
                    },
                    "transformer_block_configs": [
                        {
                            'dropout': True,
                            'iterations': 3,
                            'num_attention_heads': 8,
                            'shifted_window': False,
                            'window_size': (2, 2)
                        },
                        {
                            'dropout': True,
                            'iterations': 3,
                            'num_attention_heads': 8,
                            'shifted_window': False,
                            'window_size': (2, 2)
                        },
                    ],
                },
                {
                    "patch_embedding_info": {
                        'patch_size': 8,
                        'in_channels': 64,
                    },
                    "transformer_block_configs": [
                        {
                            'dropout': True,
                            'iterations': 3,
                            'num_attention_heads': 8,
                            'shifted_window': False,
                            'window_size': (2, 2)
                        },
                        {
                            'dropout': True,
                            'iterations': 3,
                            'num_attention_heads': 8,
                            'shifted_window': False,
                            'window_size': (2, 2)
                        },
                    ],
                },
            ],
        )
        self.encoder = Encoder.from_config(self.config)

    def test_transformer_blocks(self) -> None:
        encoder = self.encoder
        transformer_blocks = encoder.transformer_blocks
        patch_embedding_configs = self.config.patch_embedding_configs

        self.assertEqual(len(transformer_blocks), 2)
        for i in range(2):
            self.assertEqual(len(transformer_blocks[i]), 5)
            for j in range(5):
                self.assertEqual(transformer_blocks[i][j].in_channels, patch_embedding_configs[j].in_channels)

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
        config = self.config
        encoder = self.encoder
        patch_embedding_configs = config.patch_embedding_configs

        # Test forward pass
        batch_size = 1
        patch_embeddings = [
            torch.rand(batch_size, patch_embedding_config.in_patches, patch_embedding_config.in_channels)
            for patch_embedding_config in patch_embedding_configs
        ]

        output = encoder(patch_embeddings)

        self.assertEqual(len(output), 5)
        for i in range(5):
            self.assertEqual(output[i].shape, patch_embeddings[i].shape)


if __name__ == '__main__':
    unittest.main()
