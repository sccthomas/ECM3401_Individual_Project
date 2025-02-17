import unittest
import unittest.mock

import matplotlib.pyplot as _plt
import torch
from PIL import Image

from src.training.visualisation import display_tensor_mask, display_tensor_image, display_attention_weights
from src.vision_transformer.model import SemanticSegmentationVisionTransformer


class TestDisplayFunctions(unittest.TestCase):
    def test_display_tensor_mask(self) -> None:
        mask = torch.rand(1, 256, 256)
        mask = (mask > 0.5).float()
        pil = display_tensor_mask(mask)
        self.assertIsInstance(pil, Image.Image)

    def test_display_tensor_image(self) -> None:
        image = torch.rand(3, 256, 256)
        pil = display_tensor_image(image)
        self.assertIsInstance(pil, Image.Image)

    def test_display_attention_weights(self) -> None:
        model = SemanticSegmentationVisionTransformer(
            image_dims=(3, 256, 256),
            num_encoder_layers=4,
            use_swin_transformer=False,
            use_heavyweight_decoder=False,
            skip_layer_ratio=2,
            use_learnable_skip_layers=True,
            use_skip_layer_gated_attention=True,
            encoder_dropout_rate=0.25,
            patch_fusion_dropout_rate=0.25,
            decoder_dropout_rate=0.25,
            num_encoder_heads=4,
            num_classes=1,
            patch_embedding_scale_1=(16, 1024),
            patch_embedding_scale_2=(8, 768),

        )
        images = torch.rand(2, 3, 256, 256)
        with unittest.mock.patch.object(_plt, 'show') as mock_show:
            display_attention_weights(
                model=model,
                img_original=images[0],
                img_pre=images[0],
                patch_sizes=[16, 8],
            )
            mock_show.assert_called()

    def test_display_attention_weights_swin_transformer_encoder(self) -> None:
        model = SemanticSegmentationVisionTransformer(
            image_dims=(3, 256, 256),
            num_encoder_layers=4,
            use_swin_transformer=True,
            use_heavyweight_decoder=False,
            skip_layer_ratio=2,
            use_learnable_skip_layers=True,
            use_skip_layer_gated_attention=False,
            encoder_dropout_rate=0.25,
            patch_fusion_dropout_rate=0.25,
            decoder_dropout_rate=0.25,
            num_encoder_heads=4,
            num_classes=1,
            patch_embedding_scale_1=(16, 1024),
            patch_embedding_scale_2=(8, 768),
        )
        images = torch.rand(2, 3, 256, 256)
        with unittest.mock.patch.object(_plt, 'show') as mock_show:
            display_attention_weights(
                model=model,
                img_original=images[0],
                img_pre=images[0],
                patch_sizes=[16, 8],
            )
            mock_show.assert_called()


if __name__ == '__main__':
    unittest.main()
