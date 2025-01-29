import abc as _abc
import typing as _t

import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _f

import src.vision_transformer.common.decoder as _decoder
import src.vision_transformer.common.patch_fusion as _patch_fusion
import src.vision_transformer.common.swin_transformer_encoder as _swin_transformer_encoder


class SemanticSegmentationVisionTransformerBase(_nn.Module):
    """
    Semantic Segmentation Vision Transformer Base Class. This class defines the base architecture for the vision
    transformer model. Child classes should define the abstract methods and use the provided methods to create the
    essential components of the model.
    """

    def __init__(
            self,
            image_dims: _t.Tuple[int, int, int],
            num_encoder_layers: int,
            patch_embedding_scales: _t.List[_t.Tuple[int, int]],
    ) -> None:
        """
        Initialize the vision_transformer.

        :param image_dims: The dimensions of the input image.
        :param num_encoder_layers: The number of encoder layers
        :param patch_embedding_scales: The patch embedding configurations for each scale.
        """
        super(SemanticSegmentationVisionTransformerBase, self).__init__()

        _, height, width = image_dims

        assert height == width, "Input image must be square."

        self.__image_dims = image_dims[1:]
        self.__num_encoder_layers = num_encoder_layers
        self.__num_patch_fusion_layers = num_encoder_layers - 1
        self.__decoder = _decoder.Decoder.create(
            patch_embedding_scales=patch_embedding_scales,
            input_dims=image_dims,
            output_dims=(1, height, width),
        )

    @property
    def image_dims(self) -> _t.Tuple[int, int]:
        """
        Get the image dimensions.

        :return: The image dimensions.
        """
        return self.__image_dims

    @property
    def num_encoder_layers(self) -> int:
        """
        Get the number of encoder layers.

        :return: The number of encoder layers.
        """
        return self.__num_encoder_layers

    @property
    def decoder(self) -> _decoder.Decoder:
        """
        Get the decoder module.

        :return: The decoder module.
        """
        return self.__decoder

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

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        """
        Forward pass of the vision transformer model. This method applies the patch embedding, encoder, and decoder
        stages to the input tensor. Each stage returns a dictionary of tensors that are passed to the next stage. These
        represent the different patch embedding scales.

        :param x: The input tensor.
        :return: The output tensor.
        """
        image_dims = self.__image_dims
        decoder = self.__decoder

        # Patch Embedding
        kwargs = self.apply_patch_embedding_stage(x)
        # Encoder Stage
        kwargs = self.apply_encoder_stage(**kwargs)
        # Decoder Stage
        x1 = decoder(kwargs)
        assert x1.shape[2:] == image_dims

        return x1

    def _create_encoder_layers_for_scale_X(
            self, embed_dim: int,
    ) -> '_nn.ModuleList[_nn.TransformerEncoderLayer]':
        """
        Create Classic Transformer encoder layers for scale X.

        :param embed_dim: Patch embedding dimension.
        :return: Module list of Transformer encoder layers.
        """
        num_encoder_layers = self.__num_encoder_layers

        kwargs = {'nhead': 16, 'dropout': 0.25, 'activation': _f.gelu}
        encoders_scale_X = _nn.ModuleList(
            [
                _nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    dim_feedforward=int(embed_dim * 2),
                    **kwargs,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        return encoders_scale_X

    def _create_swin_encoder_layers_for_scale_X(
            self, *, H: int, embed_dim: int, input_resolution: _t.Tuple[int, int]
    ) -> '_nn.ModuleList[_swin_transformer_encoder.SwinTransformerBlock]':
        """
        Create Swin Transformer encoder layers for scale X.

        :param H:  Height of the input tensor after patch embedding.
        :param embed_dim: Patch embedding dimension.
        :param input_resolution: The resolution of the input tensor.
        :return: Module list of Swin Transformer encoder layers.
        """
        num_encoder_layers = self.__num_encoder_layers

        window_size = max(H // 4, 4)
        shift_size = window_size // 2
        kwargs = {
            'num_heads': 16,
            'drop': 0.25,
            'attn_drop': 0.25,
            'drop_path': 0.25,
        }
        encoders_scale_X = _nn.ModuleList(
            [
                _swin_transformer_encoder.SwinTransformerBlock(
                    dim=embed_dim,
                    input_resolution=input_resolution,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else shift_size,
                    **kwargs,
                )
                for i in range(num_encoder_layers)
            ]
        )

        return encoders_scale_X

    def _create_patch_fusion_layers_for_scale_X_to_Y(
            self, *, in_embed: int, out_embed: int
    ) -> '_nn.ModuleList[_patch_fusion.PatchFusion]':
        """
        Create patch fusion layers for scale X to Y.

        :param in_embed: In patch embedding dimension.
        :param out_embed: Out patch embedding dimension.
        :return: Module list of patch fusion layers.
        """
        num_patch_fusion_layers = self.__num_patch_fusion_layers

        kwargs = {
            'in_embed': in_embed,
            'out_embed': out_embed,
        }
        patch_fusion_layers_scale_X_to_Y = _nn.ModuleList(
            [
                _patch_fusion.PatchFusion(
                    **kwargs,
                )
                for _ in range(num_patch_fusion_layers)
            ]
        )

        return patch_fusion_layers_scale_X_to_Y
