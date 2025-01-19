import unittest

import torch

from src.vision_transformer.common.decoder import Decoder


class TestDecoder(unittest.TestCase):
    def test_forward(self) -> None:
        patch_embedding = torch.randn(1, 1024, 768)

        decoder = Decoder.create(1024, 768, (1, 512, 512))

        output = decoder(patch_embedding)

        self.assertEqual(output.shape, (1, 1, 512, 512))

        patch_embedding = torch.randn(1, 64, 1024)

        decoder = Decoder.create(64, 1024, (1, 256, 256))

        output = decoder(patch_embedding)

        self.assertEqual(output.shape, (1, 1, 256, 256))


if __name__ == '__main__':
    unittest.main()
