import typing as _t

import torch as _torch
import torch.nn as _nn


class Decoder(_nn.Module):
    """
    Decoder module that will upsample the final patch embedding to the output dimensions.
    """

    def __init__(
            self,
            up_sample_to_common_scale_convs: _t.Dict[str, _nn.Module],
            final_embedding_up_sample_convs: _nn.Sequential,
            prediction_head: _nn.Module,
    ) -> None:
        """

        :param up_sample_to_common_scale_convs: Transposed convolutions to upsample the embeddings to a common scale.
        :param final_embedding_up_sample_convs: Transposed convolutions to upsample the embeddings.
        :param prediction_head: Prediction head to predict the output.
        """

        super(Decoder, self).__init__()
        self.__up_sample_to_common_scale_convs = up_sample_to_common_scale_convs
        self.__final_embedding_up_sample_convs = final_embedding_up_sample_convs
        self.__prediction_head = prediction_head

        self.__initialize_weights()

    @classmethod
    def create(
            cls,
            patch_embedding_scales: _t.List[_t.Tuple[int, int]],
            input_dims: _t.Tuple[int, int, int],
            output_dims: _t.Tuple[int, int, int]
    ) -> 'Decoder':
        """
        Create a decoder that will upsample the final embeddings to the output dimensions.

        :param patch_embedding_scales:
        :param output_dims:
        :return:
        """
        final_patch_size = patch_embedding_scales[-1][0]
        final_embed_dim = patch_embedding_scales[-1][1]

        up_sample_to_common_scale_convs = {}
        for i, patch_embedding_dim in enumerate(patch_embedding_scales[:-1], start=1):
            patch_size = patch_embedding_dim[0]
            embed_dim = patch_embedding_dim[1]
            scale_factor = patch_size // final_patch_size
            up_sample_to_common_scale_convs[f"x{i}"] = (
                _nn.ConvTranspose2d(embed_dim, final_embed_dim, kernel_size=scale_factor, stride=scale_factor)
            )

        # Compute the number of operations required to reach the final resolution
        resolution_ = input_dims[1] // final_patch_size
        num_classes, H, W = output_dims
        num_operations = 0
        while resolution_ < H:
            resolution_ *= 2
            num_operations += 1

        # Compute the best factor to reduce the final embedding dimension
        best_factor = final_embed_dim // num_operations
        while final_embed_dim % best_factor != 0:
            best_factor += 1

        # Compute the dimensions of the transposed convolutions
        dim = final_embed_dim
        transposed_dims = [dim]
        for i in range(num_operations):
            dim_ = dim - best_factor
            if dim_ <= 0:
                dim //= 2
            else:
                dim = dim_
            transposed_dims.append(dim)

        # Create the transposed convolutions
        final_embedding_up_sample_convs = _nn.Sequential()
        for i in range(len(transposed_dims) - 1):
            dim_1 = transposed_dims[i]
            dim_2 = transposed_dims[i + 1]
            final_embedding_up_sample_convs.append(
                _nn.ConvTranspose2d(dim_1, dim_2, kernel_size=2, stride=2)
            )
            final_embedding_up_sample_convs.append(_nn.ReLU())

        # Add the final transposed convolution to predict the number of classes and a ReLU activation
        prediction_head = _nn.Conv2d(transposed_dims[-1], num_classes, kernel_size=1, stride=1)

        return cls(
            up_sample_to_common_scale_convs=up_sample_to_common_scale_convs,
            final_embedding_up_sample_convs=final_embedding_up_sample_convs,
            prediction_head=prediction_head
        )

    @property
    def prediction_head(self) -> _nn.Module:
        """
        Get the prediction head of the decoder

        :return: Prediction head.
        """
        return self.__prediction_head

    def forward(self, patch_embeddings: _t.Dict[str, _torch.Tensor]) -> _torch.Tensor:
        """
        Forward pass of the decoder to up sample and apply prediction head to a patch embedding tensor.

        :param patch_embeddings: Patch embeddings to up sample.
        :return: Predicted output tensor.
        """
        prediction_head = self.__prediction_head

        # Upsample to the final resolution
        final_embedding = self.forward_(patch_embeddings)

        # Apply the prediction head
        x = prediction_head(final_embedding)

        return x

    def forward_(self, patch_embeddings: _t.Dict[str, _torch.Tensor]) -> _torch.Tensor:
        """
        Forward pass of the decoder to up sample and apply prediction head to a patch embedding tensor.

        :param patch_embeddings: Patch embeddings to up sample.
        :return: The final embedding tensor.
        """
        up_sample_to_common_scale_convs = self.__up_sample_to_common_scale_convs
        final_embedding_up_sample_convs = self.__final_embedding_up_sample_convs
        prediction_head = self.__prediction_head

        # Reshape the final patch embedding into a 2D tensor
        final_embedding = list(patch_embeddings.values())[-1]
        B, N, C = final_embedding.shape
        H = W = int(N ** 0.5)
        final_embedding = final_embedding.reshape(B, C, H, W)

        # Fuse the patch embeddings to a common scale
        for key, conv in up_sample_to_common_scale_convs.items():
            patch_embedding = patch_embeddings[key]
            # - Reshape into a 2D tensor
            B, N, C = patch_embedding.shape
            H = W = int(N ** 0.5)
            patch_embedding = patch_embedding.reshape(B, C, H, W)
            # - Upsample to a common scale
            final_embedding = final_embedding + conv(patch_embedding)

        # Upsample to the final resolution
        final_embedding = final_embedding_up_sample_convs(final_embedding)

        return final_embedding

    def __initialize_weights(self) -> None:
        """
        Initialize the weights of the decoder.
        """
        final_embedding_up_sample_convs = self.__final_embedding_up_sample_convs
        up_sample_to_common_scale_convs = self.__up_sample_to_common_scale_convs
        prediction_head = self.__prediction_head

        for layer in final_embedding_up_sample_convs:
            if isinstance(layer, _nn.ConvTranspose2d):
                _nn.init.xavier_uniform_(layer.weight)
                _nn.init.constant_(layer.bias, 0)

        for layer in up_sample_to_common_scale_convs.values():
            _nn.init.xavier_uniform_(layer.weight)
            _nn.init.constant_(layer.bias, 0)

        _nn.init.xavier_uniform_(prediction_head.weight)
        _nn.init.constant_(prediction_head.bias, 0)
