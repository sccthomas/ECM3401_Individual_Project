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

    @property
    def convolutional_operation_weights(self) -> _t.Dict[str, _t.List[_torch.Tensor]]:
        """
        Get the weights of the convolutional operations in the decoder.

        :return: List of weights of the convolutional operations in the decoder.
        """
        patch_embedding_operations = self._patch_embedding_operations
        fused_embedding_operations = self._fused_embedding_operations

        weights = {
            'patch_embedding_operations': [
                m.weight
                for module in patch_embedding_operations.values() for m in module.modules()
                if isinstance(m, _nn.Conv2d) or isinstance(m, _nn.ConvTranspose2d)
            ],
            'fused_embedding_operations': [
                module.weight
                for module in fused_embedding_operations
                if isinstance(module, _nn.Conv2d) or isinstance(module, _nn.ConvTranspose2d)
            ],
        }

        return weights

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

        # Apply the patch embedding operations
        feature_maps = []
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
        target_patch_size, target_channels = (dim // 2 for dim in highest_resolution_patch_embedding)

        patch_embedding_operations = _nn.ModuleDict(
            {
                f'x{i}': _nn.Sequential(
                    # Block 1
                    _nn.ConvTranspose2d(
                        in_channels=embed_dim,
                        out_channels=embed_dim - (embed_dim - target_channels) // 2,
                        kernel_size=(patch_size // target_patch_size) // 2,
                        stride=(patch_size // target_patch_size) // 2,
                    ),
                    _nn.BatchNorm2d(embed_dim - (embed_dim - target_channels) // 2),
                    _nn.ReLU(),
                    _nn.Dropout2d(p=dropout_rate),

                    # Block 2
                    _nn.ConvTranspose2d(
                        in_channels=embed_dim - (embed_dim - target_channels) // 2,
                        out_channels=target_channels,
                        kernel_size=2,
                        stride=2,
                    ),
                    _nn.BatchNorm2d(target_channels),
                    _nn.ReLU(),
                    _nn.Dropout2d(p=dropout_rate),
                )
                for i, (patch_size, embed_dim) in enumerate(patch_embedding_scales, start=1)
            }
        )

        hidden_dim_1 = target_channels // 2
        hidden_dim_2 = target_channels // 4
        kernel_size = stride = target_patch_size // 2
        fused_embedding_operations = _nn.Sequential(
            # Block 1
            _nn.ConvTranspose2d(
                in_channels=target_channels * num_scales, out_channels=target_channels, kernel_size=1, stride=1
            ),
            _nn.BatchNorm2d(target_channels),
            _nn.ReLU(),
            _nn.Dropout2d(p=dropout_rate),

            # Block 2
            _nn.ConvTranspose2d(
                in_channels=target_channels, out_channels=hidden_dim_1, kernel_size=kernel_size, stride=stride
            ),
            _nn.BatchNorm2d(hidden_dim_1),
            _nn.ReLU(),
            _nn.Dropout2d(p=dropout_rate),

            # Block 3
            _nn.ConvTranspose2d(
                in_channels=hidden_dim_1, out_channels=hidden_dim_2, kernel_size=2, stride=2
            ),
        )
        prediction_head = _nn.Conv2d(in_channels=hidden_dim_2, out_channels=num_classes, kernel_size=1, stride=1)

        return cls(
            patch_embedding_operations=patch_embedding_operations,
            fused_embedding_operations=fused_embedding_operations,
            prediction_head=prediction_head
        )


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
        target_patch_size, target_channels = (dim // 2 for dim in highest_resolution_patch_embedding)

        patch_embedding_operations = _nn.ModuleDict(
            {
                f'x{i}': _nn.Sequential(
                    # Block 1
                    _nn.Conv2d(
                        in_channels=embed_dim,
                        out_channels=embed_dim - (embed_dim - target_channels) // 2,
                        kernel_size=1,
                        stride=1
                    ),
                    _nn.BatchNorm2d(embed_dim - (embed_dim - target_channels) // 2),
                    _nn.ReLU(),
                    _nn.Dropout2d(p=dropout_rate),
                    _nn.Upsample(scale_factor=(patch_size // target_patch_size) // 2, mode='nearest'),

                    # Block 2
                    _nn.Conv2d(
                        in_channels=embed_dim - (embed_dim - target_channels) // 2,
                        out_channels=target_channels,
                        kernel_size=1,
                        stride=1
                    ),
                    _nn.BatchNorm2d(target_channels),
                    _nn.ReLU(),
                    _nn.Dropout2d(p=dropout_rate),
                    _nn.Upsample(scale_factor=2, mode='nearest'),
                )
                for i, (patch_size, embed_dim) in enumerate(patch_embedding_scales, start=1)
            }
        )
        hidden_dim_1 = target_channels // 2
        hidden_dim_2 = target_channels // 4
        kernel_size = stride = target_patch_size // 2
        fused_embedding_operations = _nn.Sequential(
            # Block 1
            _nn.Conv2d(in_channels=target_channels * num_scales, out_channels=target_channels, kernel_size=1, stride=1),
            _nn.BatchNorm2d(target_channels),
            _nn.ReLU(),
            _nn.Dropout2d(p=dropout_rate),

            # Block 2
            _nn.ConvTranspose2d(
                in_channels=target_channels, out_channels=hidden_dim_1, kernel_size=kernel_size, stride=stride
            ),

            # Block 3
            _nn.ConvTranspose2d(
                in_channels=hidden_dim_1, out_channels=hidden_dim_2, kernel_size=2, stride=2
            ),
        )
        prediction_head = _nn.Conv2d(in_channels=hidden_dim_2, out_channels=num_classes, kernel_size=1, stride=1)

        return cls(
            patch_embedding_operations=patch_embedding_operations,
            fused_embedding_operations=fused_embedding_operations,
            prediction_head=prediction_head
        )
