import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import torch

from src.self_supervised_learning.contrastive_loss import ContrastivePreTraining, visualize_tsne
from src.vision_transformer.model import SemanticSegmentationVisionTransformer


class TestContrastivePreTraining(unittest.TestCase):
    def test_forward(self) -> None:
        model = SemanticSegmentationVisionTransformer(
            image_dims=(3, 128, 128),
            num_encoder_layers=4,
            use_swin_transformer=False,
            use_heavyweight_decoder=False,
            skip_layer_ratio=4,
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
        x = torch.rand(2, 3, 128, 128).float()

        contrastive_model = ContrastivePreTraining(
            model=model,
            encoder_dims=[1024, 768],
            projection_dim=128,
            pooling_method='max',
        )

        y = contrastive_model.forward_loss(x)

        self.assertEqual(y.shape, torch.Size([]))


class TestTSNE(unittest.TestCase):
    def test_plot(self) -> None:
        model = SemanticSegmentationVisionTransformer(
            image_dims=(3, 128, 128),
            num_encoder_layers=4,
            use_swin_transformer=False,
            use_heavyweight_decoder=False,
            skip_layer_ratio=4,
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
        x = torch.rand(2, 3, 128, 128).float()

        contrastive_model = ContrastivePreTraining(
            model=model,
            encoder_dims=[1024, 768],
            projection_dim=128,
            pooling_method='max',
        )

        with patch.object(plt, 'show') as mock_show:
            visualize_tsne(
                model=contrastive_model,
                images=x,
            )
            mock_show.assert_called()


if __name__ == '__main__':
    unittest.main()
