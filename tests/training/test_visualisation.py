import unittest
import unittest.mock

import matplotlib.pyplot as _plt
import torch
from PIL import Image

from src.training.visualisation import display_tensor_mask, display_tensor_image, display_attention_weights
from src.vision_transformer.model.two_scales import SemanticSegmentationVisionTransformer


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

    def test_display_attention_weights(self) -> None:
        model = SemanticSegmentationVisionTransformer(
            image_dims=(3, 256, 256),
            num_encoder_layers=12,
            patch_embedding_scale_1=(16, 1024),
            patch_embedding_scale_2=(8, 768),
        )
        images = torch.rand(2, 3, 256, 256)
        for patch_size, scale_key in [(16, 'x1'), (8, 'x2')]:
            with unittest.mock.patch.object(_plt, 'show') as mock_show:
                display_attention_weights(
                    model=model,
                    img_original=images[0],
                    img_pre=images[0],
                    patch_size=patch_size,
                    scale_key=scale_key,
                    layer=0
                )
                mock_show.assert_called()


if __name__ == '__main__':
    unittest.main()
