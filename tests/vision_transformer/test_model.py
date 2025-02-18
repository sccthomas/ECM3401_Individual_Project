import unittest

import torch

from src.vision_transformer.model import SemanticSegmentationVisionTransformer


class TestSemanticSegmentationVisionTransformer(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("mps" if torch.cuda.is_available() else "cpu")

    def test_single_scale(self) -> None:
        device = self.device
        model = SemanticSegmentationVisionTransformer(
            image_dims=(3, 128, 128),
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
            patch_embedding_scale_1=(32, 1024),
        ).to(device)

        # Test the forward pass
        x = torch.rand(2, 3, 128, 128).float().to(device)
        y = model(x)

        self.assertEqual(y.shape, (2, 1, 128, 128))
        self.assertFalse(torch.isnan(y).any())

        # Test the forward pass with the keep_attention_scores flag
        x = torch.rand(2, 3, 128, 128).float().to(device)
        y = model(x, keep_attention_scores=True)
        attention_scores = model.get_attention_scores()

        self.assertEqual(y.shape, (2, 1, 128, 128))
        self.assertFalse(torch.isnan(y).any())
        self.assertEqual(len(attention_scores), 1)
        for attention_score in attention_scores.values():
            self.assertEqual(len(attention_score), 2)

    def test_two_scales(self) -> None:
        device = self.device
        model = SemanticSegmentationVisionTransformer(
            image_dims=(3, 128, 128),
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
            patch_embedding_scale_1=(32, 1024),
            patch_embedding_scale_2=(16, 768),
        ).to(device)

        # Test the forward pass
        x = torch.rand(2, 3, 128, 128).float().to(device)
        y = model(x)

        self.assertEqual(y.shape, (2, 1, 128, 128))
        self.assertFalse(torch.isnan(y).any())

        # Test the forward pass with the keep_attention_scores flag
        x = torch.rand(2, 3, 128, 128).float().to(device)
        y = model(x, keep_attention_scores=True)
        attention_scores = model.get_attention_scores()

        self.assertEqual(y.shape, (2, 1, 128, 128))
        self.assertFalse(torch.isnan(y).any())
        self.assertEqual(len(attention_scores), 2)
        for attention_score in attention_scores.values():
            self.assertEqual(len(attention_score), 2)

    def test_three_scales(self) -> None:
        device = self.device
        model = SemanticSegmentationVisionTransformer(
            image_dims=(3, 128, 128),
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
            patch_embedding_scale_1=(32, 1024),
            patch_embedding_scale_2=(16, 768),
            patch_embedding_scale_3=(8, 512),
        ).to(device)

        # Test the forward pass
        x = torch.rand(2, 3, 128, 128).float().to(device)
        y = model(x)

        self.assertEqual(y.shape, (2, 1, 128, 128))
        self.assertFalse(torch.isnan(y).any())

        # Test the forward pass with the keep_attention_scores flag
        x = torch.rand(2, 3, 128, 128).float().to(device)
        y = model(x, keep_attention_scores=True)
        attention_scores = model.get_attention_scores()

        self.assertEqual(y.shape, (2, 1, 128, 128))
        self.assertFalse(torch.isnan(y).any())
        self.assertEqual(len(attention_scores), 3)
        for attention_score in attention_scores.values():
            self.assertEqual(len(attention_score), 2)

    def test_four_scales(self) -> None:
        device = self.device
        model = SemanticSegmentationVisionTransformer(
            image_dims=(3, 128, 128),
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
            patch_embedding_scale_1=(32, 1024),
            patch_embedding_scale_2=(16, 768),
            patch_embedding_scale_3=(8, 512),
            patch_embedding_scale_4=(4, 256),
        ).to(device)

        # Test the forward pass
        x = torch.rand(2, 3, 128, 128).float().to(device)
        y = model(x)

        self.assertEqual(y.shape, (2, 1, 128, 128))
        self.assertFalse(torch.isnan(y).any())

        # Test the forward pass with the keep_attention_scores flag
        x = torch.rand(2, 3, 128, 128).float().to(device)
        y = model(x, keep_attention_scores=True)
        attention_scores = model.get_attention_scores()

        self.assertEqual(y.shape, (2, 1, 128, 128))
        self.assertFalse(torch.isnan(y).any())
        self.assertEqual(len(attention_scores), 4)
        for attention_score in attention_scores.values():
            self.assertEqual(len(attention_score), 2)

    def test_five_scales(self) -> None:
        device = self.device
        model = SemanticSegmentationVisionTransformer(
            image_dims=(3, 128, 128),
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
            patch_embedding_scale_1=(32, 1024),
            patch_embedding_scale_2=(16, 768),
            patch_embedding_scale_3=(8, 512),
            patch_embedding_scale_4=(4, 256),
            patch_embedding_scale_5=(2, 128),
        ).to(device)

        # Test the forward pass
        x = torch.rand(2, 3, 128, 128).float().to(device)
        y = model(x)

        self.assertEqual(y.shape, (2, 1, 128, 128))
        self.assertFalse(torch.isnan(y).any())

        # Test the forward pass with the keep_attention_scores flag
        x = torch.rand(2, 3, 128, 128).float().to(device)
        y = model(x, keep_attention_scores=True)
        attention_scores = model.get_attention_scores()

        self.assertEqual(y.shape, (2, 1, 128, 128))
        self.assertFalse(torch.isnan(y).any())
        self.assertEqual(len(attention_scores), 5)
        for attention_score in attention_scores.values():
            self.assertEqual(len(attention_score), 2)


if __name__ == '__main__':
    unittest.main()
