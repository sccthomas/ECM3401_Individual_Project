import abc as _abc
import typing as _t

import torch as _torch
import torch.nn as _nn


# This module is not part of the public API
class BaseDecoder(_nn.Module):
    """
    Base class for the decoder module of the Vision Transformer.
    """

    def __init__(
            self,
            patch_embedding_operations: _nn.ModuleDict,
            fused_embedding_operations: _nn.Sequential,
            prediction_head: _nn.Module,
    ) -> None:
        """

        :param patch_embedding_operations: Dictionary containing the patch embedding operations for each scale
        :param fused_embedding_operations: Sequence of operations for the fused patch embeddings.
        :param prediction_head: Prediction head for the decoder.
        """
        super(BaseDecoder, self).__init__()
        self._patch_embedding_operations = patch_embedding_operations
        self._fused_embedding_operations = fused_embedding_operations
        self._prediction_head = prediction_head

        self.__initialize_weights()

    @classmethod
    @_abc.abstractmethod
    def create(
            cls,
            patch_embedding_scales: _t.List[_t.Tuple[int, int]],
            input_dims: _t.Tuple[int, int, int],
            output_dims: _t.Tuple[int, int, int],
            dropout_rate: float,
    ) -> 'BaseDecoder':
        """
        Create a decoder module.

        :param patch_embedding_scales: List of tuples containing the patch size and embedding dimension.
        :param input_dims: Input dimensions of the image.
        :param output_dims: Output dimensions of the image.
        :param dropout_rate: Dropout rate.
        :return: Decoder module.
        """

    @property
    def prediction_head(self) -> _nn.Module:
        """

        :return: The prediction head of the decoder.
        """
        return self._prediction_head

    @_abc.abstractmethod
    def forward(
            self,
            patch_embeddings: _t.Dict[str, _torch.Tensor],
            apply_prediction_head: bool = True,
    ) -> _torch.Tensor:
        """
        Forward pass of the decoder.

        :param patch_embeddings: Patch embeddings to up sample.
        :param apply_prediction_head: Whether to apply the prediction head.
        :return: Predicted output tensor.
        """

    def __initialize_weights(self) -> None:
        """
        Initialize the weights of the decoder.
        """
        patch_embedding_operations = self._patch_embedding_operations
        fused_embedding_operations = self._fused_embedding_operations
        prediction_head = self._prediction_head

        for module in fused_embedding_operations:
            if isinstance(module, _nn.Conv2d):
                _nn.init.kaiming_normal_(module.weight)
                _nn.init.constant_(module.bias, 0)
            elif isinstance(module, _nn.BatchNorm2d):
                _nn.init.constant_(module.weight, 1)
                _nn.init.constant_(module.bias, 0)

        for module in patch_embedding_operations.values():
            for m in module.modules():
                if isinstance(m, _nn.Conv2d):
                    _nn.init.kaiming_normal_(m.weight)
                    _nn.init.constant_(m.bias, 0)
                elif isinstance(m, _nn.BatchNorm2d):
                    _nn.init.constant_(m.weight, 1)
                    _nn.init.constant_(m.bias, 0)

        _nn.init.kaiming_normal_(prediction_head.weight)
        _nn.init.constant_(prediction_head.bias, 0)


class HeavyWeightDecoder(BaseDecoder):
    """
    Heavy-weight decoder module that will upsample the final patch embedding to the output dimensions.
    - This decoder will use transposed convolutions to upsample the final patch embedding to the output dimensions.
    - The number of transposed convolutions will be determined by the number of operations required to reach the final
        resolution.
    - The final transposed convolution will predict the number of classes.
    """

    def __init__(
            self,
            patch_embedding_operations: _nn.ModuleDict,
            fused_embedding_operations: _nn.Sequential,
            prediction_head: _nn.Module,
    ) -> None:
        """
        Initialize the heavy-weight decoder.

        :param patch_embedding_operations: The patch embedding operations.
        :param fused_embedding_operations: The fused embedding operations.
        :param prediction_head: The prediction head.
        """
        super(HeavyWeightDecoder, self).__init__(
            patch_embedding_operations=patch_embedding_operations,
            fused_embedding_operations=fused_embedding_operations,
            prediction_head=prediction_head
        )

    @classmethod
    def create(
            cls,
            patch_embedding_scales: _t.List[_t.Tuple[int, int]],
            input_dims: _t.Tuple[int, int, int],
            output_dims: _t.Tuple[int, int, int],
            dropout_rate: float,
    ) -> 'HeavyWeightDecoder':
        """
        Create a heavy-weight decoder that will upsample the final embeddings to the output dimensions.

        :param patch_embedding_scales: List of tuples containing the patch size and embedding dimension.
        :param input_dims: Input dimensions of the image.
        :param output_dims: Output dimensions of the image.
        :param dropout_rate: Dropout rate.
        :return: Decoder module.
        """
        final_patch_size = patch_embedding_scales[-1][0]
        final_embed_dim = patch_embedding_scales[-1][1]

        patch_embedding_operations = _nn.ModuleDict()
        for i, patch_embedding_dim in enumerate(patch_embedding_scales[:-1], start=1):
            patch_size = patch_embedding_dim[0]
            embed_dim = patch_embedding_dim[1]
            scale_factor = patch_size // final_patch_size
            patch_embedding_operations[f"x{i}"] = _nn.ModuleList(
                [
                    _nn.ConvTranspose2d(embed_dim, final_embed_dim, kernel_size=scale_factor, stride=scale_factor),
                    _nn.BatchNorm2d(final_embed_dim),
                    _nn.ReLU(),
                    _nn.Dropout2d(dropout_rate),
                ]
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
        fused_embedding_operations = _nn.Sequential()
        for i in range(len(transposed_dims) - 1):
            dim_1 = transposed_dims[i]
            dim_2 = transposed_dims[i + 1]
            fused_embedding_operations.extend(
                _nn.Sequential(
                    _nn.ConvTranspose2d(dim_1, dim_2, kernel_size=2, stride=2),
                    _nn.BatchNorm2d(dim_2),
                    _nn.ReLU(),
                    _nn.Dropout2d(dropout_rate),
                )
            )

        # Add the final transposed convolution to predict the number of classes and a ReLU activation
        prediction_head = _nn.Conv2d(transposed_dims[-1], num_classes, kernel_size=1, stride=1)

        return cls(
            patch_embedding_operations=patch_embedding_operations,
            fused_embedding_operations=fused_embedding_operations,
            prediction_head=prediction_head
        )

    def forward(
            self,
            patch_embeddings: _t.Dict[str, _torch.Tensor],
            apply_prediction_head: bool = True,
    ) -> _torch.Tensor:
        """
        Forward pass of the decoder to up sample and apply prediction head to a patch embedding tensor.

        :param patch_embeddings: Patch embeddings to up sample.
        :param apply_prediction_head: Whether to apply the prediction head.
        :return: Predicted output tensor.
        """
        patch_embedding_operations = self._patch_embedding_operations
        fused_embedding_operations = self._fused_embedding_operations
        prediction_head = self._prediction_head

        # Reshape the final patch embedding into a 2D tensor
        final_embedding = list(patch_embeddings.values())[-1]
        B, N, C = final_embedding.shape
        H = W = int(N ** 0.5)
        final_embedding = final_embedding.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # Fuse the patch embeddings to a common scale
        for key, (conv, norm, relu, dropout) in patch_embedding_operations.items():
            patch_embedding = patch_embeddings[key]
            # - Reshape into a 2D tensor
            B, N, C = patch_embedding.shape
            H = W = int(N ** 0.5)
            patch_embedding = patch_embedding.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            # - Upsample to a common scale
            #   - Convolution
            patch_embedding = conv(patch_embedding)
            #   - Batch normalization
            patch_embedding = norm(patch_embedding)
            #   - ReLU activation
            final_embedding = relu(final_embedding + patch_embedding)
            #   - Dropout
            final_embedding = dropout(final_embedding)

        # Upsample to the final resolution
        x = fused_embedding_operations(final_embedding)

        # Apply the prediction head
        if apply_prediction_head:
            x = prediction_head(x)

        return x


class LightWeightDecoder(BaseDecoder):
    """
    Light-weight decoder module that will upsample the final patch embedding to the output dimensions.
    - This decoder will use nearest neighbor up sampling to upsample the final patch embedding to the output dimensions.
    - The number of nearest neighbor up sampling will be determined by the number of operations required to reach the
        final resolution.
    - The final nearest neighbor up sampling will predict the number of classes
    """

    def __init__(
            self,
            patch_embedding_operations: _nn.ModuleDict,
            fused_embedding_operations: _nn.Sequential,
            prediction_head: _nn.Module,
    ) -> None:
        """
        Initialize the light-weight decoder.

        :param patch_embedding_operations: The patch embedding operations.
        :param fused_embedding_operations: The fused embedding operations.
        :param prediction_head: The prediction head.
        """
        super(LightWeightDecoder, self).__init__(
            patch_embedding_operations=patch_embedding_operations,
            fused_embedding_operations=fused_embedding_operations,
            prediction_head=prediction_head
        )

    @classmethod
    def create(
            cls,
            patch_embedding_scales: _t.List[_t.Tuple[int, int]],
            input_dims: _t.Tuple[int, int, int],
            output_dims: _t.Tuple[int, int, int],
            dropout_rate: float,
    ) -> 'LightWeightDecoder':
        """
        Create a light-weight decoder that will upsample the final embeddings to the output dimensions.

        :param patch_embedding_scales: List of tuples containing the patch size and embedding dimension.
        :param input_dims: Input dimensions of the image.
        :param output_dims: Output dimensions of the image.
        :param dropout_rate: Dropout rate.
        :return: Decoder module.
        """
        H, W = input_dims[1:]
        num_classes = output_dims[0]
        num_scales = len(patch_embedding_scales)
        highest_resolution_patch_embedding = min(patch_embedding_scales, key=lambda x: x[0])
        out_patch_size, out_channels = highest_resolution_patch_embedding

        patch_embedding_scales.remove(highest_resolution_patch_embedding)

        patch_embedding_operations = _nn.ModuleDict(
            {
                f'x{i}': _nn.Sequential(
                    _nn.Conv2d(in_channels=embed_dim, out_channels=out_channels, kernel_size=1, stride=1),
                    _nn.BatchNorm2d(out_channels),
                    _nn.ReLU(),
                    _nn.Dropout(p=dropout_rate),
                    _nn.Upsample(scale_factor=patch_size / out_patch_size, mode='nearest'),
                )
                for i, (patch_size, embed_dim) in enumerate(patch_embedding_scales, start=1)
            }
        )
        hidden_dim = out_channels // 2
        fused_embedding_operations = _nn.Sequential(
            _nn.Conv2d(in_channels=out_channels * num_scales, out_channels=out_channels, kernel_size=1, stride=1),
            _nn.BatchNorm2d(out_channels),
            _nn.ReLU(),
            _nn.Dropout(p=dropout_rate),
            _nn.ConvTranspose2d(
                in_channels=out_channels, out_channels=hidden_dim, kernel_size=out_patch_size, stride=out_patch_size
            ),
        )
        prediction_head = _nn.Conv2d(in_channels=hidden_dim, out_channels=num_classes, kernel_size=1, stride=1)

        return cls(
            patch_embedding_operations=patch_embedding_operations,
            fused_embedding_operations=fused_embedding_operations,
            prediction_head=prediction_head
        )

    def forward(
            self,
            patch_embeddings: _t.Dict[str, _torch.Tensor],
            apply_prediction_head: bool = True,
    ) -> _torch.Tensor:
        """
        Forward pass of the decoder to up sample and apply prediction head to a patch embedding tensor.

        :param patch_embeddings: Patch embeddings to up sample.
        :param apply_prediction_head: Whether to apply the prediction head.
        :return: Predicted output tensor.
        """
        patch_embedding_operations = self._patch_embedding_operations
        fused_embedding_operations = self._fused_embedding_operations
        prediction_head = self._prediction_head

        key = set(
            patch_embeddings.keys()).difference(
            set(patch_embedding_operations.keys())
        ).pop()
        patch_embedding = patch_embeddings[key]
        B, P, E = patch_embedding.shape
        P = int(P ** 0.5)
        patch_embedding = patch_embedding.reshape(B, P, P, E).permute(0, 3, 1, 2).contiguous()

        # Apply the patch embedding operations
        feature_maps = [patch_embedding, ]
        for key, operation in patch_embedding_operations.items():
            patch_embedding = patch_embeddings[key]
            B, P, E = patch_embedding.shape
            P = int(P ** 0.5)
            patch_embedding = patch_embedding.reshape(B, P, P, E).permute(0, 3, 1, 2).contiguous()
            feature_maps.append(operation(patch_embedding))

        # Fuse the patch embeddings
        x = fused_embedding_operations(_torch.cat(feature_maps, dim=1))

        # Apply the prediction head
        if apply_prediction_head:
            x = prediction_head(x)

        return x
