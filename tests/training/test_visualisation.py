import unittest
import unittest.mock

import matplotlib.pyplot as _plt
import torch
from PIL import Image

from src.training.visualisation import display_tensor_mask, display_tensor_image, display_overlaid_avg_attention


class TestDisplayFunctions(unittest.TestCase):
    def test_display_tensor_mask(self) -> None:
        mask = torch.rand(1, 256, 256)
        mask = (mask > 0.5).float()
        with unittest.mock.patch.object(Image.Image, 'show') as mock_show:
            display_tensor_mask(mask)
            mock_show.assert_called_once()

    def test_display_tensor_image(self) -> None:
        image = torch.rand(3, 256, 256)
        with unittest.mock.patch.object(Image.Image, 'show') as mock_show:
            img = display_tensor_image(image)
            mock_show.assert_called_once()

    def test_display_overlaid_avg_attention(self) -> None:
        weights = torch.rand(12, 256, 256)
        images = torch.rand(2, 3, 256, 256)
        with unittest.mock.patch.object(_plt, 'show') as mock_show:
            display_overlaid_avg_attention(weights, images, image_idx=0, alpha=0.5)
            mock_show.assert_called_once()


if __name__ == '__main__':
    unittest.main()
