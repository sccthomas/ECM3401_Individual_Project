import unittest

import torch

from src.vision_transformer.common.swin_transformer_encoder import Mlp, window_partition, window_reverse, \
    WindowAttention, SwinTransformerBlock


class TestSwinTransformer(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 2
        self.height = 8
        self.width = 8
        self.channels = 4
        self.window_size = 4
        self.num_heads = 2
        self.dim = self.channels

    def test_mlp_forward(self) -> None:
        mlp = Mlp(in_features=self.channels, hidden_features=8, out_features=self.channels)
        x = torch.randn(self.batch_size, self.channels)
        output = mlp(x)
        self.assertEqual(output.shape, (self.batch_size, self.channels))

    def test_window_partition(self) -> None:
        x = torch.randn(self.batch_size, self.height, self.width, self.channels)
        windows = window_partition(x, self.window_size)
        expected_shape = (
            (self.height // self.window_size) * (self.width // self.window_size) * self.batch_size,
            self.window_size,
            self.window_size,
            self.channels,
        )
        self.assertEqual(windows.shape, expected_shape)

    def test_window_reverse(self) -> None:
        num_windows = (self.height // self.window_size) * (self.width // self.window_size)
        windows = torch.randn(num_windows * self.batch_size, self.window_size, self.window_size, self.channels)
        x = window_reverse(windows, self.window_size, self.height, self.width)
        self.assertEqual(x.shape, (self.batch_size, self.height, self.width, self.channels))

    def test_window_attention_forward(self) -> None:
        attention = WindowAttention(
            dim=self.dim,
            window_size=(self.window_size, self.window_size),
            num_heads=self.num_heads
        )
        x = torch.randn(self.batch_size * (self.height // self.window_size) * (self.width // self.window_size),
                        self.window_size * self.window_size,
                        self.channels)
        output = attention(x)
        self.assertEqual(output.shape, x.shape)

    def test_swin_transformer_block(self) -> None:
        block = SwinTransformerBlock(
            dim=self.dim,
            input_resolution=(self.height, self.width),
            num_heads=self.num_heads,
            window_size=self.window_size,
            shift_size=2,
            mlp_ratio=4.0
        )
        x = torch.randn(self.batch_size, self.height * self.width, self.channels)
        output = block(x)
        self.assertEqual(output.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
