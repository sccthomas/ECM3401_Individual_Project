import collections as _col
import typing as _t


class ModelConfig:
    def __init__(self, patch_embedding_dims: _t.Iterable[_t.Dict[str, int]], encoder_config: _t.Dict[str, int]) -> None:
        self.__encoder_config = _Encoder_Config(**encoder_config)
        self.__patch_embedding_dims = [
            _Patch_Embedding_Dim(**patch_embedding_dim)
            for patch_embedding_dim in patch_embedding_dims
        ]

    @property
    def encoder(self) -> '_Encoder_Config':
        return self.__encoder_config

    @property
    def patch_embedding_dims(self) -> _t.Iterable['_Patch_Embedding_Dim']:
        return self.__patch_embedding_dims


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Private Helpers
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


_Patch_Embedding_Dim = _col.namedtuple('_Patch_Embedding_Dim', ['patch_len', 'vector_len'])
_Encoder_Config = _col.namedtuple('_Encoder_Config', ['num_stages', 'iterations'])
