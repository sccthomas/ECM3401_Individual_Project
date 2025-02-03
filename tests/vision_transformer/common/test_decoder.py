import unittest

import torch

from src.vision_transformer.common.decoder import Decoder


class TestDecoder(unittest.TestCase):
    def test_forward(self) -> None:
        patch_embeddings = {
            'x1': torch.randn(1, 16, 1024),
            'x2': torch.randn(1, 64, 768),
            'x3': torch.randn(1, 256, 512),
        }

        decoder = Decoder.create(
            patch_embedding_scales=[(32, 1024), (16, 768), (8, 512)],
            input_dims=(3, 128, 128),
            output_dims=(1, 128, 128),
            dropout_rate=0.25,
        )

        output = decoder(patch_embeddings)

        self.assertEqual(output.shape, (1, 1, 128, 128))


if __name__ == '__main__':
    unittest.main()
