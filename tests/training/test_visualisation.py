import unittest
import unittest.mock

import matplotlib.pyplot as plt
import torch

from src.training.visualisation import display_tensor_mask, display_tensor_image, display_attention_weights


class TestDisplayFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("mps" if torch.cuda.is_available() else "cpu")

    def test_display_tensor_mask(self) -> None:
        device = self.device

        mask = torch.rand(1, 256, 256).to(device)
        mask = (mask > 0.5).float()
        with unittest.mock.patch.object(plt, 'show') as mock_show:
            display_tensor_mask(mask)
            mock_show.assert_called()

    def test_display_tensor_image(self) -> None:
        device = self.device

        image = torch.rand(3, 256, 256).to(device)
        with unittest.mock.patch.object(plt, 'show') as mock_show:
            display_tensor_image(image)
            mock_show.assert_called()

    def test_display_attention_weights(self) -> None:
        device = self.device

        # Normal transformer encoder
        attention_scores = {
            ('x1', 16): {
                "encoder": [torch.rand(1, 4, 256, 256), torch.rand(1, 4, 256, 256)],
                "patch_fusion": [torch.rand(1, 4, 256, 256), torch.rand(1, 4, 256, 256)],
            }
        }

        images = torch.rand(2, 3, 256, 256).to(device)
        with unittest.mock.patch.object(plt, 'show') as mock_show:
            display_attention_weights(
                image=images[0],
                attention_scores=attention_scores,
            )
            mock_show.assert_called()

        # Swin transformer encoder
        attention_scores = {
            ('x1', 16): {
                "encoder": [torch.rand(16, 4, 16, 16), torch.rand(16, 4, 16, 16)],
                "patch_fusion": [torch.rand(1, 4, 256, 256), torch.rand(1, 4, 256, 256)],
            }
        }

        images = torch.rand(2, 3, 256, 256).to(device)
        with unittest.mock.patch.object(plt, 'show') as mock_show:
            display_attention_weights(
                image=images[0],
                attention_scores=attention_scores,
            )
            mock_show.assert_called()


if __name__ == '__main__':
    unittest.main()
