import typing as _t


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Model Config
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class ModelConfig:
    """
    Model config class that defines encoder, decoder and patch embedding configurables and behaviours.
    """

    def __init__(self, config: _t.Dict[str, _t.Any]) -> None:
        """

        :param config: A config dictionary containing all configurable information.
        """
        self.__encoder = EncoderConfig(num_stages=config['num_stages'], patch_embedding_configs=config['encoder'])

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

    def __init__(self, *, num_stages: int, patch_embedding_configs: _t.List[_t.Dict[str, _t.Any]]) -> None:
        """

        :param num_stages: Number of stages in the encoder stage.
        :param patch_embedding_configs: Patch embedding configs, specifying the patch size and feed forward network.
        """
        self.__num_stages = num_stages
        self.__patch_embedding_configs = [
            PatchEmbeddingConfig(**patch_embedding_config)
            for patch_embedding_config in patch_embedding_configs
        ]

    @property
    def num_stages(self) -> int:
        """

        :return: The number of stages.
        """
        return self.__num_stages

    @property
    def patch_embedding_configs(self) -> _t.List['PatchEmbeddingConfig']:
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


class PatchEmbeddingConfig:
    """
    Patch Embedding config.
    Details:
    - Patch size.
    - Feed forward ViT network and parameters.
    """

    def __init__(
            self,
            *,
            patch_embedding_info: _t.Dict[str, int],
            transformer_block_configs: _t.List[_t.Dict[str, int]],
    ) -> None:
        """

        :param patch_embedding_info: Patch Embedding size information.
        :param transformer_block_configs: Feed forward transformer information and parameters.
        """
        num_patches = patch_embedding_info['num_patches']
        vector_len = patch_embedding_info['vector_len']
        kwargs = {"num_patches": num_patches, "vector_len": vector_len}
        self.__num_patches = num_patches
        self.__vector_len = vector_len
        self.__transformer_block_configs = [
            TransformerBlockConfig(**kwargs, **transformer_block_config)
            for transformer_block_config in transformer_block_configs
        ]

    @property
    def num_patches(self) -> int:
        """

        :return: The number of Patch Embeddings.
        """
        return self.__num_patches

    @property
    def vector_len(self) -> int:
        """

        :return: The length of each Patch Embedding.
        """
        return self.__vector_len

    @property
    def transformer_block_configs(self) -> _t.List['TransformerBlockConfig']:
        """

        :return: List of transformer block configs.
        """
        return self.__transformer_block_configs


class TransformerBlockConfig:
    """
    Transformer block config class containing all ViT parameters.
    """

    def __init__(
            self,
            dropout: bool,
            iterations: int,
            num_heads: int,
            num_patches: int,
            shifted_window: bool,
            vector_len: int,
            window_size: _t.Tuple[int, int],
    ) -> None:
        """

        :param dropout: Include dropout.
        :param iterations: Iterations inside ViT block.
        :param num_heads: Number of heads during attention.
        :param num_patches: The number of patches to process.
        :param shifted_window: Shift windows.
        :param vector_len: Patch vector length.
        :param window_size: Window size.
        """
        self.__dropout = dropout
        self.__iterations = iterations
        self.__num_heads = num_heads
        self.__num_patches = num_patches
        self.__shifted_window = shifted_window
        self.__vector_len = vector_len
        self.__window_size = window_size

    @property
    def vector_len(self) -> int:
        """

        :return: Patch vector length.
        """
        return self.__vector_len

    @property
    def num_patches(self) -> int:
        """

        :return: Number of patches.
        """
        return self.__num_patches

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
    def num_heads(self) -> int:
        """

        :return: Number of heads.
        """
        return self.__num_heads

    @property
    def iterations(self) -> int:
        """

        :return: Iterations.
        """
        return self.__iterations
