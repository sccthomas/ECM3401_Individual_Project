import unittest

import torch

from src.model.single_scale.encoder import TransformerEncoder


class TestTransformerEncoder(unittest.TestCase):
    def test_forward(self) -> None:
        device = torch.device('mps')
        patch_embedding = torch.Tensor(10, 256, 768).to(device)

        transformer_encoder = TransformerEncoder(embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1,
                                                 num_layers=6).to(device)
        output = transformer_encoder(patch_embedding)

        self.assertEqual(output.shape, (10, 256, 768))


if __name__ == '__main__':
    unittest.main()
