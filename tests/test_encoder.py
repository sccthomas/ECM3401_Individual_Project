from model.encoder import Encoder
from model.config import ModelConfig
import unittest
from torch.nn import ModuleList, Linear
import torch


class TestEncoder(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(
            patch_embedding_dims=[
                {'patch_len': 16, 'vector_len': 1024},
                {'patch_len': 64, 'vector_len': 768},
                {'patch_len': 256, 'vector_len': 512},
                {'patch_len': 1024, 'vector_len': 256},
                {'patch_len': 4096, 'vector_len': 64},
            ],
            encoder_config={'num_stages': 3, 'iterations': 1}
        )
        self.encoder = Encoder(self.config)

    def test_transformer_blocks(self) -> None:
        encoder = self.encoder
        transformer_blocks = encoder.transformer_blocks
        patch_embedding_dims = self.config.patch_embedding_dims

        self.assertEqual(len(transformer_blocks), 3)
        for i in range(3):
            self.assertEqual(len(transformer_blocks[i]), 5)
            for j in range(5):
                self.assertEqual(transformer_blocks[i][j].vector_len, patch_embedding_dims[j].vector_len)

    def test_skip_connections(self) -> None:
        encoder = self.encoder
        skip_connections = encoder.skip_connections

        self.assertEqual(len(skip_connections), 3)

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

        for i in range(3):
            self.assertEqual(len(skip_connections[i]), 5)
            for j in range(5):
                actual_skip_connection_scales = [
                    (linear_operation.in_features, linear_operation.out_features)
                    for linear_operation in skip_connections[i][j].linear_operations
                ]
                self.assertEqual(actual_skip_connection_scales, expected_skip_connections_scales[j])

    def test_forward(self) -> None:
        encoder = self.encoder

        # Test forward pass
        batch_size = 1
        patch_embeddings = [
            torch.rand(batch_size, 16, 1024),
            torch.rand(batch_size, 64, 768),
            torch.rand(batch_size, 256, 512),
            torch.rand(batch_size, 1024, 256),
            torch.rand(batch_size, 4096, 64),
        ]

        output = encoder(patch_embeddings)

        self.assertEqual(len(output), 5)
        for i in range(5):
            self.assertEqual(output[i].shape, patch_embeddings[i].shape)
