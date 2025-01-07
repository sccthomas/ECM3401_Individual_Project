import unittest

import torch

from src.model import ModelConfig
from src.model.model import SemanticSegmentationVisionTransformer


class TestSemanticSegmentationVisionTransformer(unittest.TestCase):
    def test_forward(self) -> None:
        config = ModelConfig.create(
            input_dimensions=(3, 512, 512),
            output_dimensions=(1, 512, 512),
            num_encoder_stages=2,
            num_classes=1,
            patch_embedding_config_dicts=[
                # Patch Embedding Config 1
                {
                    "patch_embedding_info": {
                        'patch_size': 128,
                        'in_channels': 1024,
                    },
                    "encoder_block_configs": [
                        {
                            'dropout': True,
                            'iterations': 3,
                            'num_attention_heads': 8,
                            'shifted_window': True,
                            'window_size': (2, 2)
                        }
                        for _ in range(3)
                    ],
                    "decoder_block_config": {
                        'dropout': True,
                        'iterations': 3,
                        'num_attention_heads': 8,
                        'shifted_window': True,
                        'window_size': (2, 2)
                    }
                },
                # Patch Embedding Config 2
                {
                    "patch_embedding_info": {
                        'patch_size': 8,
                        'in_channels': 64,
                    },
                    "encoder_block_configs": [
                        {
                            'dropout': True,
                            'iterations': 3,
                            'num_attention_heads': 8,
                            'shifted_window': True,
                            'window_size': (16, 16)
                        }
                        for _ in range(3)
                    ],
                    "decoder_block_config": {
                        'dropout': True,
                        'iterations': 3,
                        'num_attention_heads': 8,
                        'shifted_window': True,
                        'window_size': (16, 16)
                    }
                },
            ],
        )
        device = torch.device('mps')
        model = SemanticSegmentationVisionTransformer.from_config(config).to(device)

        # Test the forward method.
        batch = torch.randn(10, 3, 512, 512).to(device)
        predictions = model(batch)

        self.assertEqual(predictions.shape, (10, 1, 512, 512))


if __name__ == '__main__':
    unittest.main()
