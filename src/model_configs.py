import src.model.config as _config


def large_model_configuration() -> _config.ModelConfig:
    patch_embedding_config_dicts = [
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
    ]
    model_config = _config.ModelConfig.create(
        input_dimensions=(3, 512, 512),
        output_dimensions=(1, 512, 512),
        num_encoder_stages=3,
        num_classes=1,
        patch_embedding_config_dicts=patch_embedding_config_dicts,
    )

    return model_config


def small_model_configuration() -> _config.ModelConfig:
    patch_embedding_config_dicts = [
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
    ]
    model_config = _config.ModelConfig.create(
        input_dimensions=(3, 512, 512),
        output_dimensions=(1, 512, 512),
        num_encoder_stages=3,
        num_classes=1,
        patch_embedding_config_dicts=patch_embedding_config_dicts,
    )

    return model_config
