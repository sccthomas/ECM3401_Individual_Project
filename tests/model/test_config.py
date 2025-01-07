import unittest

from src.model import ModelConfig, EncoderConfig, DecoderConfig, TransformerBlockConfig, PatchEmbeddingConfig


class TestModelConfig(unittest.TestCase):
    def test_create(self) -> None:
        model_config = ModelConfig.create(
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
            ],
        )

        self.assertEqual(model_config.input_dimensions, (3, 512, 512))
        self.assertEqual(model_config.output_dimensions, (1, 512, 512))
        self.assertEqual(model_config.num_classes, 1)

        patch_embedding_config = model_config.patch_embedding_configs[0]
        self.assertEqual(patch_embedding_config.in_patches, 16)
        self.assertEqual(patch_embedding_config.in_channels, 1024)
        self.assertEqual(patch_embedding_config.patch_size, 128)
        self.assertEqual(patch_embedding_config.patch_resolution, (4, 4))

        encoder_config = model_config.encoder_config
        self.assertEqual(encoder_config.num_stages, 2)
        self.assertEqual(len(encoder_config.transformer_block_configs), 1)
        self.assertEqual(len(encoder_config.patch_embedding_configs), 1)

        transformer_block_configs = encoder_config.transformer_block_configs[0]
        self.assertEqual(len(transformer_block_configs), 2)

        transformer_block_config = transformer_block_configs[0]
        self.assertEqual(transformer_block_config.iterations, 3)
        self.assertEqual(transformer_block_config.num_attention_heads, 8)
        self.assertEqual(transformer_block_config.window_size, (2, 2))
        self.assertEqual(transformer_block_config.shifted_window, False)
        self.assertEqual(transformer_block_config.dropout, False)

        transformer_block_config = transformer_block_configs[1]
        self.assertEqual(transformer_block_config.iterations, 3)
        self.assertEqual(transformer_block_config.num_attention_heads, 8)
        self.assertEqual(transformer_block_config.window_size, (2, 2))
        self.assertEqual(transformer_block_config.shifted_window, False)
        self.assertEqual(transformer_block_config.dropout, True)

        decoder_config = model_config.decoder_config
        self.assertEqual(decoder_config.num_classes, 1)
        self.assertEqual(decoder_config.output_dimensions, (1, 512, 512))
        self.assertEqual(decoder_config.max_in_channels, 1024)
        self.assertEqual(len(decoder_config.transformer_block_configs), 1)
        self.assertEqual(len(decoder_config.patch_embedding_configs), 1)

        transformer_block_config = decoder_config.transformer_block_configs[0]
        self.assertEqual(transformer_block_config.iterations, 3)
        self.assertEqual(transformer_block_config.num_attention_heads, 8)
        self.assertEqual(transformer_block_config.window_size, (2, 2))
        self.assertEqual(transformer_block_config.shifted_window, False)
        self.assertEqual(transformer_block_config.dropout, True)


class TestEncoderConfig(unittest.TestCase):
    def test_something(self) -> None:
        encoder_confiig = EncoderConfig(
            num_stages=2,
            transformer_block_configs=[
                [
                    TransformerBlockConfig(
                        iterations=3,
                        num_attention_heads=8,
                        window_size=(2, 2),
                        shifted_window=False,
                        dropout=False,
                    ),
                    TransformerBlockConfig(
                        iterations=3,
                        num_attention_heads=8,
                        window_size=(2, 2),
                        shifted_window=False,
                        dropout=True,
                    ),
                ],
            ],
            patch_embedding_configs=[
                PatchEmbeddingConfig(
                    in_patches=16,
                    in_channels=1024,
                    patch_resolution=(4, 4),
                    patch_size=128,
                ),
            ]
        )

        self.assertEqual(encoder_confiig.num_stages, 2)
        self.assertEqual(len(encoder_confiig.transformer_block_configs), 1)
        self.assertEqual(len(encoder_confiig.patch_embedding_configs), 1)

        transformer_block_configs = encoder_confiig.transformer_block_configs[0]
        self.assertEqual(len(transformer_block_configs), 2)

        transformer_block_config = transformer_block_configs[0]
        self.assertEqual(transformer_block_config.iterations, 3)
        self.assertEqual(transformer_block_config.num_attention_heads, 8)
        self.assertEqual(transformer_block_config.window_size, (2, 2))
        self.assertEqual(transformer_block_config.shifted_window, False)
        self.assertEqual(transformer_block_config.dropout, False)

        transformer_block_config = transformer_block_configs[1]
        self.assertEqual(transformer_block_config.iterations, 3)
        self.assertEqual(transformer_block_config.num_attention_heads, 8)
        self.assertEqual(transformer_block_config.window_size, (2, 2))
        self.assertEqual(transformer_block_config.shifted_window, False)
        self.assertEqual(transformer_block_config.dropout, True)

        patch_embedding_config = encoder_confiig.patch_embedding_configs[0]
        self.assertEqual(patch_embedding_config.in_patches, 16)
        self.assertEqual(patch_embedding_config.in_channels, 1024)
        self.assertEqual(patch_embedding_config.patch_resolution, (4, 4))
        self.assertEqual(patch_embedding_config.patch_size, 128)


class TestDecoderConfig(unittest.TestCase):
    def test_something(self) -> None:
        decoder_config = DecoderConfig(
            num_classes=1,
            output_dimensions=(3, 512, 512),
            max_in_channels=1024,
            transformer_block_configs=[
                TransformerBlockConfig(
                    iterations=2,
                    num_attention_heads=8,
                    window_size=(2, 2),
                    shifted_window=True,
                    dropout=False,
                ),
            ],
            patch_embedding_configs=[
                PatchEmbeddingConfig(
                    in_patches=16,
                    in_channels=1024,
                    patch_resolution=(4, 4),
                    patch_size=128,
                ),
            ],
        )

        self.assertEqual(decoder_config.num_classes, 1)
        self.assertEqual(decoder_config.output_dimensions, (3, 512, 512))
        self.assertEqual(decoder_config.max_in_channels, 1024)
        self.assertEqual(len(decoder_config.transformer_block_configs), 1)
        self.assertEqual(len(decoder_config.patch_embedding_configs), 1)

        transformer_block_config = decoder_config.transformer_block_configs[0]
        self.assertEqual(transformer_block_config.iterations, 2)
        self.assertEqual(transformer_block_config.num_attention_heads, 8)
        self.assertEqual(transformer_block_config.window_size, (2, 2))
        self.assertEqual(transformer_block_config.shifted_window, True)
        self.assertEqual(transformer_block_config.dropout, False)

        patch_embedding_config = decoder_config.patch_embedding_configs[0]
        self.assertEqual(patch_embedding_config.in_patches, 16)
        self.assertEqual(patch_embedding_config.in_channels, 1024)
        self.assertEqual(patch_embedding_config.patch_resolution, (4, 4))
        self.assertEqual(patch_embedding_config.patch_size, 128)


class TestTransformerBlockConfig(unittest.TestCase):
    def test(self) -> None:
        transformer_block_config = TransformerBlockConfig(
            iterations=3,
            num_attention_heads=8,
            window_size=(4, 4),
            shifted_window=True,
            dropout=True,
        )

        self.assertEqual(transformer_block_config.iterations, 3)
        self.assertEqual(transformer_block_config.num_attention_heads, 8)
        self.assertEqual(transformer_block_config.window_size, (4, 4))
        self.assertEqual(transformer_block_config.shifted_window, True)
        self.assertEqual(transformer_block_config.dropout, True)


class TestPatchEmbeddingConfig(unittest.TestCase):
    def test_create(self) -> None:
        patch_embedding_config = PatchEmbeddingConfig.create(
            input_dimensions=(3, 512, 512),
            patch_size=128,
            in_channels=1024
        )

        self.assertEqual(patch_embedding_config.in_patches, 16)
        self.assertEqual(patch_embedding_config.in_channels, 1024)
        self.assertEqual(patch_embedding_config.patch_size, 128)
        self.assertEqual(patch_embedding_config.patch_resolution, (4, 4))


if __name__ == '__main__':
    unittest.main()
