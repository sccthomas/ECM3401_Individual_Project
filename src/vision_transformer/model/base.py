import abc as _abc
import typing as _t

import torch as _torch
import torch.nn as _nn


class SemanticSegmentationVisionTransformerBase(_nn.Module):
    """
    Semantic Segmentation Vision Transformer Base Class.
    """

    def __init__(
            self,
            image_dims: _t.Tuple[int, int, int],
    ) -> None:
        """
        Initialize the vision_transformer.

        :param image_dims: The dimensions of the input image.
        """
        super(SemanticSegmentationVisionTransformerBase, self).__init__()

        _, height, width = image_dims

        assert height == width, "Input image must be square."

        self.__image_dims = image_dims[1:]

    @_abc.abstractmethod
    def apply_patch_embedding_stage(self, x: _torch.Tensor) -> _t.Dict[str, _torch.Tensor]:
        """
        Apply the patch embedding to the input tensor.

        :param x: The input tensor.
        :return: The patch embeddings for each scale.
        """

    @_abc.abstractmethod
    def apply_encoder_stage(self, **kwargs) -> _t.Dict[str, _torch.Tensor]:
        """
        Apply the encoder stage to the input tensors.

        :return: Encoded tensors for each scale.
        """

    @_abc.abstractmethod
    def apply_decoder_stage(self, **kwargs) -> _torch.Tensor:
        """
        Apply the decoder stage to the input tensors.

        :return: The decoded tensor.
        """

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        """
        Forward pass.

        :param x: The input tensor.
        :return: The output tensor.
        """
        image_dims = self.__image_dims

        # Patch Embedding
        kwargs = self.apply_patch_embedding_stage(x)
        # Encoder Stage
        kwargs = self.apply_encoder_stage(**kwargs)
        # Decoder Stage
        x1 = self.apply_decoder_stage(**kwargs)

        assert x1.shape[2:] == image_dims

        return x1
