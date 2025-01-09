import typing as _t

import numpy as _np


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Model Config
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class ModelConfig:
    """
    Model config class that defines encoder, decoder and patch embedding configurables and behaviours.
    """

    def __init__(
            self,
            *,
            input_dimensions: _t.Tuple[int, int, int],
            output_dimensions: _t.Tuple[int, int, int],
            num_classes: int,
            patch_embedding_configs,
            encoder_config,
            decoder_config,
    ) -> None:
        """

        :param input_dimensions: Input dimensions, (B, C, H, W).
        :param output_dimensions: Output dimensions, (B, C, H, W).
        :param num_classes: Number of classes.
        :param patch_embedding_configs: Patch embedding configurations.
        :param encoder_config: Encoder configuration.
        :param decoder_config: Decoder configuration.
        """
        self.__input_dimensions = input_dimensions
        self.__output_dimensions = output_dimensions
        self.__num_classes = num_classes
        self.__patch_embedding_configs = patch_embedding_configs
        self.__encoder_config = encoder_config
        self.__decoder_config = decoder_config

    @classmethod
    def create(
            cls,
            input_dimensions: _t.Tuple[int, int, int],
            output_dimensions: _t.Tuple[int, int, int],
            num_encoder_stages: int,
            num_classes: int,
            patch_embedding_config_dicts: _t.List[
                _t.Dict[
                    str, _t.Union[
                        _t.Dict[str, _t.Union[int, bool]],
                        _t.List[
                            _t.Dict[
                                str, _t.Union[int, bool]
                            ]
                        ]
                    ]
                ]
            ],
    ) -> 'ModelConfig':
        """
        Create model configuration from dictionary and input parameters.

        :param input_dimensions: Input image dimensions. (B, C, H, W)
        :param output_dimensions: Output image dimensions. (B, C, H, W)
        :param num_encoder_stages: Number of encoder stages.
        :param num_classes: Number of classes.
        :param patch_embedding_config_dicts: List of dictionaries containing patch embedding and transformer block
               configurations.
        :return: Model configuration instance.
        """
        # Create `PatchEmbeddingConfig` and `TransformerBlockConfig` instances.
        encoder_transformer_block_configs = []
        decoder_transformer_block_configs = []
        patch_embedding_configs = []
        max_in_channels = 0
        for patch_embedding_config_dict in patch_embedding_config_dicts:
            # - Patch Embedding Config
            patch_embedding_info = patch_embedding_config_dict["patch_embedding_info"]
            patch_embedding_configs.append(
                PatchEmbeddingConfig.create(input_dimensions, **patch_embedding_info)
            )
            max_in_channels = max(max_in_channels, patch_embedding_info["in_channels"])
            # - Encoder Transformer Block Configs
            encoder_transformer_block_configs.append(
                [
                    TransformerBlockConfig(**transformer_block_config)
                    for transformer_block_config in patch_embedding_config_dict["encoder_block_configs"]
                ]
            )
            # - Decoder Transformer Block Config
            decoder_transformer_block_configs.append(
                TransformerBlockConfig(**patch_embedding_config_dict["decoder_block_config"])
            )

        # Create `EncoderConfig` and `DecoderConfig` instances.
        encoder_config = EncoderConfig(
            num_stages=num_encoder_stages,
            transformer_block_configs=encoder_transformer_block_configs,
            patch_embedding_configs=patch_embedding_configs
        )
        decoder_config = DecoderConfig(
            num_classes=num_classes,
            output_dimensions=output_dimensions,
            max_in_channels=max_in_channels,
            transformer_block_configs=decoder_transformer_block_configs,
            patch_embedding_configs=patch_embedding_configs
        )

        return cls(
            input_dimensions=input_dimensions,
            output_dimensions=output_dimensions,
            num_classes=num_classes,
            patch_embedding_configs=patch_embedding_configs,
            encoder_config=encoder_config,
            decoder_config=decoder_config,
        )

    @property
    def input_dimensions(self) -> _t.Tuple[int, int, int]:
        """

        :return: Input Dimensions.
        """
        return self.__input_dimensions

    @property
    def output_dimensions(self) -> _t.Tuple[int, int, int]:
        """

        :return: Output Dimensions.
        """
        return self.__output_dimensions

    @property
    def num_classes(self) -> int:
        """

        :return: Number of Encoder Stages
        """
        return self.__num_classes

    @property
    def patch_embedding_configs(self) -> _t.List['PatchEmbeddingConfig']:
        """

        :return: Patch Embedding Configs
        """
        return self.__patch_embedding_configs

    @property
    def encoder_config(self) -> 'EncoderConfig':
        """

        :return: Encoder Config
        """
        return self.__encoder_config

    @property
    def decoder_config(self) -> 'DecoderConfig':
        """

        :return: Decoder Config
        """
        return self.__decoder_config


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Encoder Config
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class EncoderConfig:
    """
    Encoder stage config class.
    """

    def __init__(
            self,
            *,
            num_stages: int,
            transformer_block_configs: _t.List[_t.List['TransformerBlockConfig']],
            patch_embedding_configs: _t.List['PatchEmbeddingConfig'],
    ) -> None:
        """

        :param num_stages: The number of stages.
        :param transformer_block_configs: The transformer block configurations.
        :param patch_embedding_configs:  The patch embedding configurations.
        """
        self.__num_stages = num_stages
        self.__transformer_block_configs = transformer_block_configs
        self.__patch_embedding_configs = patch_embedding_configs

    @property
    def num_stages(self) -> int:
        """

        :return: The number of stages.
        """
        return self.__num_stages

    @property
    def transformer_block_configs(self) -> _t.List[_t.List['TransformerBlockConfig']]:
        """

        :return: List of transformer block configs.
        """
        return self.__transformer_block_configs

    @property
    def patch_embedding_configs(self) -> _t.List['PatchEmbeddingConfig']:
        """

        :return: List of patch embedding configs.
        """
        return self.__patch_embedding_configs


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Decoder Config
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class DecoderConfig:
    """
    Decoder stage config class.
    """

    def __init__(
            self,
            *,
            num_classes: int,
            output_dimensions: _t.Tuple[int, int, int],
            max_in_channels: int,
            transformer_block_configs: _t.List['TransformerBlockConfig'],
            patch_embedding_configs: _t.List['PatchEmbeddingConfig'],
    ) -> None:
        """

        :param num_classes: The number of classes.
        :param output_dimensions: The output dimensions.
        :param max_in_channels: The maximum number of input channels.
        :param transformer_block_configs: The transformer block configurations.
        :param patch_embedding_configs: The patch embedding configurations.
        """
        self.__num_classes = num_classes
        self.__output_dimensions = output_dimensions
        self.__max_in_channels = max_in_channels
        self.__transformer_block_configs = list(reversed(transformer_block_configs))
        self.__patch_embedding_configs = sorted(patch_embedding_configs, key=lambda x: x.in_channels)

    @property
    def num_classes(self) -> int:
        """

        :return: The number of classes.
        """
        return self.__num_classes

    @property
    def output_dimensions(self) -> _t.Tuple[int, int, int]:
        """

        :return: The output dimensions.
        """
        return self.__output_dimensions

    @property
    def max_in_channels(self) -> int:
        """

        :return: The maximum number of input channels.
        """
        return self.__max_in_channels

    @property
    def transformer_block_configs(self) -> _t.List['TransformerBlockConfig']:
        """

        :return: List of transformer block configs.
        """
        return self.__transformer_block_configs

    @property
    def patch_embedding_configs(self) -> _t.List['PatchEmbeddingConfig']:
        """

        :return: List of patch embedding configs.
        """
        return self.__patch_embedding_configs


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Transformer Block Config
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class TransformerBlockConfig:
    """
    Transformer block config class containing all ViT parameters.
    """

    def __init__(
            self,
            iterations: int,
            num_attention_heads: int,
            window_size: _t.Tuple[int, int],
            shifted_window: bool,
            dropout: bool,
    ) -> None:
        """

        :param iterations: Iterations inside ViT block.
        :param num_attention_heads: Number of heads during attention.
        :param window_size: Window size.
        :param shifted_window: Shift windows.
        :param dropout: Include dropout.
        """
        self.__iterations = iterations
        self.__num_attention_heads = num_attention_heads
        self.__window_size = window_size
        self.__shifted_window = shifted_window
        self.__dropout = dropout

    @property
    def window_size(self) -> _t.Tuple[int, int]:
        """

        :return: Window size.
        """
        return self.__window_size

    @property
    def shifted_window(self) -> bool:
        """

        :return: Shifted window.
        """
        return self.__shifted_window

    @property
    def dropout(self) -> bool:
        """

        :return: Dropout.
        """
        return self.__dropout

    @property
    def num_attention_heads(self) -> int:
        """

        :return: Number of attention heads.
        """
        return self.__num_attention_heads

    @property
    def iterations(self) -> int:
        """

        :return: Number of iterations.
        """
        return self.__iterations


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Patch Embedding Config
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class PatchEmbeddingConfig:
    """
    Patch Embedding config class.
    """

    def __init__(
            self,
            *,
            in_patches: int,
            in_channels: int,
            patch_resolution: _t.Tuple[int, int],
            patch_size: int,
    ) -> None:
        self.__in_patches = in_patches
        self.__in_channels = in_channels
        self.__patch_resolution = patch_resolution
        self.__patch_size = patch_size

    @classmethod
    def create(
            cls,
            input_dimensions: _t.Tuple[int, int, int],
            patch_size: int,
            in_channels: int,
    ) -> 'PatchEmbeddingConfig':
        patch_resolution = tuple([int(dim // patch_size) for dim in input_dimensions[1:]])
        in_patches = int(_np.prod(patch_resolution))

        return cls(
            in_patches=in_patches,
            in_channels=in_channels,
            patch_resolution=patch_resolution,
            patch_size=patch_size,
        )

    @property
    def patch_resolution(self) -> _t.Tuple[int, int]:
        """

        :return: The resolution of the image after patching.
        """
        return self.__patch_resolution

    @property
    def patch_size(self) -> int:
        """

        :return: The patch size.
        """
        return self.__patch_size

    @property
    def in_patches(self) -> int:
        """

        :return: The number of Patch Embeddings.
        """
        return self.__in_patches

    @property
    def in_channels(self) -> int:
        """

        :return: The length of each Patch Embedding.
        """
        return self.__in_channels
