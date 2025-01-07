import unittest

import torch as _torch
import torch.nn as _nn
import torch.optim as _optim
import torch.utils.data as _data

import dataset.snow as _snow
import model.config as _config
import model.model as _model
from main import train_model


class TestMain(unittest.TestCase):
    def test_train_model(self) -> None:
        # Create model
        model_config = _config.ModelConfig.create(
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
                        'patch_size': 64,
                        'in_channels': 768,
                    },
                    "encoder_block_configs": [
                        {
                            'dropout': True,
                            'iterations': 3,
                            'num_attention_heads': 8,
                            'shifted_window': True,
                            'window_size': (4, 4)
                        }
                        for _ in range(3)
                    ],
                    "decoder_block_config": {
                        'dropout': True,
                        'iterations': 3,
                        'num_attention_heads': 8,
                        'shifted_window': True,
                        'window_size': (4, 4)
                    }
                },
                # Patch Embedding Config 3
                {
                    "patch_embedding_info": {
                        'patch_size': 32,
                        'in_channels': 512,
                    },
                    "encoder_block_configs": [
                        {
                            'dropout': True,
                            'iterations': 3,
                            'num_attention_heads': 8,
                            'shifted_window': True,
                            'window_size': (4, 4)
                        }
                        for _ in range(3)
                    ],
                    "decoder_block_config": {
                        'dropout': True,
                        'iterations': 3,
                        'num_attention_heads': 8,
                        'shifted_window': True,
                        'window_size': (4, 4)
                    }
                },
                # Patch Embedding Config 4
                {
                    "patch_embedding_info": {
                        'patch_size': 16,
                        'in_channels': 256,
                    },
                    "encoder_block_configs": [
                        {
                            'dropout': True,
                            'iterations': 3,
                            'num_attention_heads': 8,
                            'shifted_window': True,
                            'window_size': (8, 8)
                        }
                        for _ in range(3)
                    ],
                    "decoder_block_config": {
                        'dropout': True,
                        'iterations': 3,
                        'num_attention_heads': 8,
                        'shifted_window': True,
                        'window_size': (8, 8)
                    }
                },
                # Patch Embedding Config 5
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
        device = _torch.device('mps')
        semantic_segmentation_model = _model.SemanticSegmentationVisionTransformer.from_config(model_config).to(device)

        # Create dataset loaders
        snow_dataset = _torch.utils.data.Subset(
            _snow.SnowDataset(dataset_dir_path='/Users/samuelthomas/Documents/University/4thYr_Final'
                                               '/ECM3401_Individual_Literature_Review_and_Project'
                                               '/SNOW_Semantic_Segmentation'
                                               '/snow_dataset'), range(10))
        training_dataset, validation_dataset = _data.random_split(snow_dataset, [0.8, 0.2])

        training_dataset_loader = _data.DataLoader(training_dataset, batch_size=8, shuffle=True)
        validation_dataset_loader = _data.DataLoader(validation_dataset, batch_size=2, shuffle=False)

        # Create Optimizer, Loss function and Device
        optimizer = _optim.Adam(semantic_segmentation_model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = _nn.BCEWithLogitsLoss().to(device)

        # Train model
        num_epochs = 1
        semantic_segmentation_model = train_model(
            semantic_segmentation_model,
            training_dataset_loader,
            validation_dataset_loader,
            optimizer,
            criterion,
            device,
            num_epochs,
        )


if __name__ == '__main__':
    unittest.main()
