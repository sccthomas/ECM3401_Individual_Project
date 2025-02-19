import unittest
import unittest.mock

import matplotlib.pyplot as plt
import torch

from src.training.visualisation import display_tensor_mask, display_tensor_image, display_attention_weights
from src.vision_transformer.model import SemanticSegmentationVisionTransformer


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

        ).to(device)
        images = torch.rand(2, 3, 256, 256).to(device)
        with unittest.mock.patch.object(plt, 'show') as mock_show:
            display_attention_weights(
                model=model,
                image=images[0],
                device=device,
                patch_sizes=[16, 8],
            )
            mock_show.assert_called()

    def test_display_attention_weights_swin_transformer_encoder(self) -> None:
        device = self.device

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
        ).to(device)
        images = torch.rand(2, 3, 256, 256).to(device)
        with unittest.mock.patch.object(plt, 'show') as mock_show:
            display_attention_weights(
                model=model,
                image=images[0],
                device=device,
                patch_sizes=[16, 8],
            )
            mock_show.assert_called()


if __name__ == '__main__':
    unittest.main()
