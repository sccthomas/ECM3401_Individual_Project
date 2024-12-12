from model.encoder import Encoder, _SkipConnections
from model.config import ModelConfig
import unittest
from torch.nn import ModuleList, Linear


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

        expected_skip_connections_scale_1 = ModuleList(
            [
                Linear(in_features=768, out_features=1024, bias=True),
                Linear(in_features=512, out_features=1024, bias=True),
                Linear(in_features=256, out_features=1024, bias=True),
                Linear(in_features=64, out_features=1024, bias=True),
            ]
        )

        expected_skip_connections_scale_2 = ModuleList(
            [
                Linear(in_features=1024, out_features=768, bias=True),
                Linear(in_features=512, out_features=768, bias=True),
                Linear(in_features=256, out_features=768, bias=True),
                Linear(in_features=64, out_features=768, bias=True),
            ]
        )

        expected_skip_connections_scale_3 = ModuleList(
            [
                Linear(in_features=1024, out_features=512, bias=True),
                Linear(in_features=768, out_features=512, bias=True),
                Linear(in_features=256, out_features=512, bias=True),
                Linear(in_features=64, out_features=512, bias=True),
            ]
        )

        expected_skip_connections_scale_4 = ModuleList(
            [
                Linear(in_features=1024, out_features=256, bias=True),
                Linear(in_features=768, out_features=256, bias=True),
                Linear(in_features=512, out_features=256, bias=True),
                Linear(in_features=64, out_features=256, bias=True),
            ]
        )

        expected_skip_connections_scale_5 = ModuleList(
            [
                Linear(in_features=1024, out_features=64, bias=True),
                Linear(in_features=768, out_features=64, bias=True),
                Linear(in_features=512, out_features=64, bias=True),
                Linear(in_features=256, out_features=64, bias=True),
            ]
        )

        for i in range(3):
            self.assertEqual(len(skip_connections[i]), 5)
            for j in range(5):
                for linear_operation in skip_connections[i][j].linear_operations:


    def test_forward(self) -> None:
        pass
