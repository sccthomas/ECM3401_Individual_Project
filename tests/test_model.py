from model.model import SemanticSegmentationVisionTransformer
from model.config import ModelConfig
import torch
import unittest


class TestSemanticSegmentationVisionTransformer(unittest.TestCase):
    def test_forward(self) -> None:
        config = ModelConfig.create(
            input_dimensions=(3, 512, 512),
            output_dimensions=(1, 512, 512),
            num_encoder_stages=2,
            num_classes=1,
            patch_embedding_config_dicts=[
                {
                    "patch_embedding_info": {
                        'patch_size': 128,
                        'in_channels': 1024,
                    },
                    "encoder_block_configs": [
                        {
                            'dropout': False,
                            'iterations': 3,
                            'num_attention_heads': 8,
                            'shifted_window': False,
                            'window_size': (2, 2)
                        },
                        {
                            'dropout': True,
                            'iterations': 3,
                            'num_attention_heads': 8,
                            'shifted_window': False,
                            'window_size': (2, 2)
                        },
                    ],
                    "decoder_block_config": {
                        'dropout': True,
                        'iterations': 3,
                        'num_attention_heads': 8,
                        'shifted_window': False,
                        'window_size': (2, 2)
                    }
                },
                {
                    "patch_embedding_info": {
                        'patch_size': 64,
                        'in_channels': 768,
                    },
                    "encoder_block_configs": [
                        {
                            'dropout': False,
                            'iterations': 3,
                            'num_attention_heads': 8,
                            'shifted_window': False,
                            'window_size': (2, 2)
                        },
                        {
                            'dropout': True,
                            'iterations': 3,
                            'num_attention_heads': 8,
                            'shifted_window': False,
                            'window_size': (2, 2)
                        },
                    ],
                    "decoder_block_config": {
                        'dropout': True,
                        'iterations': 3,
                        'num_attention_heads': 8,
                        'shifted_window': False,
                        'window_size': (2, 2)
                    }
                },
            ],
        )
        model = SemanticSegmentationVisionTransformer.from_config(config)

        # Test the forward method.
        batch = torch.randn(10, 3, 512, 512)
        predictions = model(batch)

        self.assertEqual(predictions.shape, (10, 1, 512, 512))


if __name__ == '__main__':
    unittest.main()
