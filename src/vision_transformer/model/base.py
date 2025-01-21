import abc as _abc
import typing as _t

import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _f

import src.vision_transformer.common.patch_fusion as _patch_fusion
import src.vision_transformer.common.swin_transformer_encoder as _swin_transformer_encoder


class SemanticSegmentationVisionTransformerBase(_nn.Module):
    """
    Semantic Segmentation Vision Transformer Base Class.
    """

    def __init__(
            self,
            image_dims: _t.Tuple[int, int, int],
            num_encoder_layers: int
    ) -> None:
        """
        Initialize the vision_transformer.

        :param image_dims: The dimensions of the input image.
        :param num_encoder_layers: The number of encoder layers
        """
        super(SemanticSegmentationVisionTransformerBase, self).__init__()

        _, height, width = image_dims

        assert height == width, "Input image must be square."

        self.__image_dims = image_dims[1:]
        self.__num_encoder_layers = num_encoder_layers
        self.__num_patch_fusion_layers = num_encoder_layers - 1

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

    def _create_swin_encoder_layers_for_scale_X(
            self, *, H: int, embed_dim: int, input_resolution: _t.Tuple[int, int], kwargs: _t.Dict[str, _t.Any]
    ) -> '_nn.ModuleList[_swin_transformer_encoder.SwinTransformerBlock]':
        """
        Create Swin Transformer encoder layers for scale X.

        :param H:  Height of the input tensor after patch embedding.
        :param embed_dim: Patch embedding dimension.
        :param input_resolution: The resolution of the input tensor.
        :param kwargs: Additional keyword arguments.
        :return: Module list of Swin Transformer encoder layers.
        """
        num_encoder_layers = self.__num_encoder_layers

        window_size = max(H // 4, 4)
        shift_size = window_size // 2

        kwargs = {'nhead': 16, 'dropout': 0.1, 'activation': _f.gelu}
        encoders_scale_X = _nn.ModuleList(
            [
                _nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    dim_feedforward=int(embed_dim * 2),
                    **kwargs,
                )
                for _ in range(num_encoder_layers)
                # _swin_transformer_encoder.SwinTransformerBlock(
                #     dim=embed_dim,
                #     input_resolution=input_resolution,
                #     window_size=window_size,
                #     shift_size=0 if (i % 2 == 0) else shift_size,
                #     **kwargs,
                # )
                for i in range(num_encoder_layers)
            ]
        )

        return encoders_scale_X

    def _create_patch_fusion_layers_for_scale_X_to_Y(
            self, *, in_patches: int, in_embed: int, out_patches: int, out_embed: int
    ) -> '_nn.ModuleList[_patch_fusion.PatchFusion]':
        """
        Create patch fusion layers for scale X to Y.

        :param in_patches: In number of patches.
        :param in_embed: In patch embedding dimension.
        :param out_patches: Out number of patches.
        :param out_embed: Out patch embedding dimension.
        :return: Module list of patch fusion layers.
        """
        num_patch_fusion_layers = self.__num_patch_fusion_layers

        kwargs = {
            'in_patches': in_patches,
            'in_embed': in_embed,
            'out_patches': out_patches,
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
