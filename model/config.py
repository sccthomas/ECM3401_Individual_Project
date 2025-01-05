import typing as _t
import numpy as _np


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Model Config
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class ModelConfig:
    """
    Model config class that defines encoder, decoder and patch embedding configurables and behaviours.
    """

    def __init__(self, input_dimensions: _t.Tuple[int, int, int], config: _t.Dict[str, _t.Any]) -> None:
        """

        :param config: A config dictionary containing all configurable information.
        """
        self.__input_dims = input_dimensions
        self.__encoder = EncoderConfig(
            input_dimensions=input_dimensions,
            num_stages=config['num_stages'],
            patch_embedding_configs=config['encoder'],
        )

    @property
    def encoder(self) -> 'EncoderConfig':
        """

        :return: The encoder config.
        """
        return self.__encoder

    @staticmethod
    def __assert_config(config: _t.Dict[str, _t.Any]) -> bool:
        """
        Static method to assert if a config is valid.

        :param config: The config being checked.
        :return: Boolean truth value.
        """
        pass


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
            input_dimensions: _t.Tuple[int, int, int],
            num_stages: int,
            patch_embedding_configs: _t.List[_t.Dict[str, _t.Any]]
    ) -> None:
        """

        :param input_dimensions: Input dimensions.
        :param num_stages: Number of stages in the encoder stage.
        :param patch_embedding_configs: Patch embedding configs, specifying the patch size and feed forward network.
        """
        self.__input_dimensions = input_dimensions
        self.__num_stages = num_stages
        self.__patch_embedding_configs = [
            PatchEmbeddingConfigEncoder(input_dimensions=input_dimensions, **patch_embedding_config)
            for patch_embedding_config in patch_embedding_configs
        ]

    @property
    def num_stages(self) -> int:
        """

        :return: The number of stages.
        """
        return self.__num_stages

    @property
    def patch_embedding_configs(self) -> _t.List['PatchEmbeddingConfigEncoder']:
        """

        :return: List of patch embedding configs.
        """
        return self.__patch_embedding_configs

    @staticmethod
    def __assert_config(config: _t.Dict[str, _t.Any]) -> bool:
        """
        Assert that an encoder config is valid.

        :param config: Encoder config.
        :return: Boolean truth value.
        """
        pass


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Decoder Config
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class DecoderConfig:
    pass


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Patch Embedding Config
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class PatchEmbeddingConfig:
    def __init__(
            self,
            *,
            input_dimensions: _t.Tuple[int, int, int],
            patch_size: int,
            in_channels: int,
    ) -> None:
        patch_resolution = tuple([int(dim // patch_size) for dim in input_dimensions[1:]])

        self.__in_patches = int(_np.prod(patch_resolution))
        self.__in_channels = in_channels
        self.__patch_resolution = patch_resolution
        self.__patch_size = patch_size

    @property
    def patch_resolution(self) -> _t.Tuple[int, int]:
        return self.__patch_resolution

    @property
    def patch_size(self) -> int:
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


class PatchEmbeddingConfigEncoder(PatchEmbeddingConfig):
    def __init__(
            self,
            *,
            input_dimensions: _t.Tuple[int, int, int],
            patch_embedding_info: _t.Dict[str, int],
            transformer_block_configs: _t.List[_t.Dict[str, int]],
    ) -> None:
        """

        :param input_dimensions: Input dimensions.
        :param patch_embedding_info: Patch Embedding size information.
        :param transformer_block_configs: Feed forward transformer information and parameters.
        """

        super(PatchEmbeddingConfigEncoder, self).__init__(
            input_dimensions=input_dimensions,
            patch_size=patch_embedding_info['patch_size'],
            in_channels=patch_embedding_info['in_channels'],
        )

        self.__transformer_block_configs = [
            TransformerBlockConfig(**transformer_block_config)
            for transformer_block_config in transformer_block_configs
        ]

    @property
    def transformer_block_configs(self) -> _t.List['TransformerBlockConfig']:
        """

        :return: List of transformer block configs.
        """
        return self.__transformer_block_configs


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

        :return: Iterations.
        """
        return self.__iterations
