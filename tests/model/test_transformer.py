import unittest

import torch
import torch.nn as nn
import torch.optim as optim

from src.model.transformer import TransformerBlockEncoder, TransformerBlockDecoder


class TestTransformerBlockEncoder(unittest.TestCase):
    def test_forward_with_training(self) -> None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        patch_embeddings = torch.rand(1, 16, 1024).to(device)
        target = torch.rand(1, 16, 1024).to(device)

        transformer_block_encoder = TransformerBlockEncoder(
            in_patches=16,
            in_channels=1024,
            patch_resolution=(4, 4),
            iterations=3,
            num_attention_heads=8,
            window_size=(2, 2),
            shifted_window=True,
            dropout=True,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(transformer_block_encoder.parameters(), lr=0.001)

        optimizer.zero_grad()
        output = transformer_block_encoder(patch_embeddings)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        self.assertEqual(output.shape, patch_embeddings.shape)
        self.assertGreater(loss.item(), 0)

        patch_embeddings = torch.rand(1, 64, 768)
        self.assertRaises(
            AssertionError,
            transformer_block_encoder,
            patch_embeddings,
        )


class TestTransformerBlockDecoder(unittest.TestCase):
    def test_forward_with_training(self) -> None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        patch_embeddings = torch.rand(1, 64, 768).to(device)
        target = torch.rand(1, 16, 1024).to(device)

        transformer_block_decoder = TransformerBlockDecoder(
            in_patches=64,
            in_channels=768,
            patch_resolution=(8, 8),
            output_dims=(16, 1024),
            iterations=3,
            num_attention_heads=8,
            window_size=(2, 2),
            shifted_window=True,
            dropout=True,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(transformer_block_decoder.parameters(), lr=0.001)

        optimizer.zero_grad()
        output = transformer_block_decoder(patch_embeddings)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        self.assertEqual(output.shape, (1, 16, 1024))
        self.assertGreater(loss.item(), 0)


if __name__ == "__main__":
    unittest.main()
