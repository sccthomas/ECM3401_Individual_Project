import unittest

import torch

from src.vision_transformer.model.three_scales import SemanticSegmentationVisionTransformer


class TestSemanticSegmentationVisionTransformer(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("mps" if torch.cuda.is_available() else "cpu")
        self.model = SemanticSegmentationVisionTransformer(
            image_dims=(3, 128, 128),
            num_encoder_layers=4,
            use_swin_transformer=False,
            use_heavyweight_decoder=False,
            skip_layer_ratio=4,
            use_skip_layer_gated_attention=True,
            encoder_dropout_rate=0.25,
            patch_fusion_dropout_rate=0.25,
            decoder_dropout_rate=0.25,
            num_encoder_heads=4,
            num_classes=1,
            patch_embedding_scale_1=(32, 1024),
            patch_embedding_scale_2=(16, 768),
            patch_embedding_scale_3=(8, 512),
        ).to(self.device)

    def test_forward(self) -> None:
        model = self.model
        device = self.device

        x = torch.rand(2, 3, 128, 128).float().to(device)

        y, _ = model(x)

        self.assertEqual(y.shape, (2, 1, 128, 128))
        self.assertFalse(torch.isnan(y).any())

    def test_forward_with_weights(self) -> None:
        model = self.model
        device = self.device

        x = torch.rand(2, 3, 128, 128).float().to(device)

        y, weights = model(x, return_attention_weights=True)

        self.assertEqual(y.shape, (2, 1, 128, 128))
        self.assertFalse(torch.isnan(y).any())

        self.assertEqual(len(weights), 3)
        self.assertEqual(len(weights["x1"]), 4)
        self.assertEqual(len(weights["x2"]), 4)
        self.assertEqual(len(weights["x3"]), 4)
        self.assertIsNotNone(weights["x1"][0])


if __name__ == '__main__':
    unittest.main()
