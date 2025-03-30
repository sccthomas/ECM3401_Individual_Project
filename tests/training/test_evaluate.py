import unittest
import unittest.mock

import matplotlib.pyplot as _plt
import torch

from src.training.evaluation import (
    evaluate_with_color_jitter,
    evaluate_with_illumination_modifications,
    evaluate_with_noise_addition,
    evaluate_with_no_modifications,
    evaluate_with_blur,
    evaluate_with_synthetic_background,
    evaluate_with_stain_variation,
)
from src.vision_transformer.model import SemanticSegmentationVisionTransformer


class TestEvaluate(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("mps" if torch.cuda.is_available() else "cpu")
        self.model = SemanticSegmentationVisionTransformer(
            image_dims=(3, 128, 128),
            num_encoder_layers=4,
            use_swin_transformer=False,
            use_heavyweight_decoder=True,
            skip_layer_ratio=2,
            use_learnable_skip_layers=True,
            use_skip_layer_gated_attention=True,
            encoder_dropout_rate=0.25,
            patch_fusion_dropout_rate=0.25,
            decoder_dropout_rate=0.25,
            num_encoder_heads=4,
            num_classes=1,
            patch_embedding_scale_1=(32, 1024),
        ).to(self.device)

    def test_evaluate_with_no_modifications(self) -> None:
        model = self.model
        device = self.device

        image = torch.rand(3, 128, 128)
        mask = torch.rand(1, 128, 128)
        with unittest.mock.patch.object(_plt, 'show') as mock_show:
            evaluate_with_no_modifications(model, image, mask, device)
            mock_show.assert_called()

    def test_evaluate_with_color_jitter(self) -> None:
        model = self.model
        device = self.device

        image = torch.rand(3, 128, 128)
        mask = torch.rand(1, 128, 128)
        with unittest.mock.patch.object(_plt, 'show') as mock_show:
            evaluate_with_color_jitter(model, image, mask, device)
            mock_show.assert_called()

    def test_evaluate_with_illumination_modifications(self) -> None:
        model = self.model
        device = self.device

        image = torch.rand(3, 128, 128)
        mask = torch.rand(1, 128, 128)
        with unittest.mock.patch.object(_plt, 'show') as mock_show:
            evaluate_with_illumination_modifications(model, image, mask, device)
            mock_show.assert_called()

    def test_evaluate_with_noise_addition(self) -> None:
        model = self.model
        device = self.device

        image = torch.rand(3, 128, 128)
        mask = torch.rand(1, 128, 128)
        with unittest.mock.patch.object(_plt, 'show') as mock_show:
            evaluate_with_noise_addition(model, image, mask, device)
            mock_show.assert_called()

    def test_evaluate_with_blur(self) -> None:
        model = self.model
        device = self.device

        image = torch.rand(3, 128, 128)
        mask = torch.rand(1, 128, 128)
        with unittest.mock.patch.object(_plt, 'show') as mock_show:
            evaluate_with_blur(model, image, mask, device)
            mock_show.assert_called()

    def test_evaluate_with_synthetic_background(self) -> None:
        model = self.model
        device = self.device

        image = torch.rand(3, 128, 128)
        mask = torch.rand(1, 128, 128)
        with unittest.mock.patch.object(_plt, 'show') as mock_show:
            evaluate_with_synthetic_background(model, image, mask, device)
            mock_show.assert_called()

    def test_evaluate_with_stain_variation(self) -> None:
        model = self.model
        device = self.device

        image = torch.rand(3, 128, 128)
        mask = torch.rand(1, 128, 128)
        with unittest.mock.patch.object(_plt, 'show') as mock_show:
            evaluate_with_stain_variation(model, image, mask, device)
            mock_show.assert_called()


if __name__ == '__main__':
    unittest.main()
