import abc as _abc
import typing as _t

import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _f

import src.vision_transformer.common.decoder as _decoder
import src.vision_transformer.common.patch_fusion as _patch_fusion
import src.vision_transformer.common.swin_transformer_encoder_layer as _swin_transformer_encoder
import src.vision_transformer.common.transformer_encoder_layer as _transformer_encoder_layer


class SemanticSegmentationVisionTransformerBase(_nn.Module):
    """
    Semantic Segmentation Vision Transformer Base Class. This class defines the base architecture for the vision
    transformer model. Child classes should define the abstract methods and use the provided methods to create the
    essential components of the model.
    """

    def __init__(
            self,
            *,
            image_dims: _t.Tuple[int, int, int],
            num_encoder_layers: int,
            decoder_type: str,
            skip_layer_ratio: int,
            patch_embedding_scales: _t.List[_t.Tuple[int, int]],
            encoder_dropout_rate: float,
            patch_fusion_dropout_rate: float,
            decoder_dropout_rate: float,
            num_encoder_heads: int,
            num_classes: int,
    ) -> None:
        """
        Initialize the vision_transformer.

        :param image_dims: The dimensions of the input image.
        :param num_encoder_layers: The number of encoder layers
        :param decoder_type: The type of decoder to use.
        :param skip_layer_ratio: The ratio of encoder layers to skip for patch fusion.
        :param patch_embedding_scales: The patch embedding configurations for each scale.
        :param encoder_dropout_rate: The dropout rate in the encoder stage.
        :param patch_fusion_dropout_rate: The dropout rate in the patch fusion stage.
        :param decoder_dropout_rate: The dropout rate in the decoder stage.
        :param num_encoder_heads: The number of encoder heads.
        :param num_classes: The number of classes.
        """
        super(SemanticSegmentationVisionTransformerBase, self).__init__()

        _, height, width = image_dims

        assert height == width, "Input image must be square."

        if decoder_type == 'lightweight':
            decoder = _decoder.LightWeightDecoder
        elif decoder_type == 'heavyweight':
            decoder = _decoder.HeavyWeightDecoder
        else:
            raise ValueError(f"Invalid decoder type: {decoder_type}")

        self.__image_dims = image_dims[1:]
        self.__num_encoder_layers = num_encoder_layers
        self.__num_patch_fusion_layers = (num_encoder_layers // skip_layer_ratio) - 1
        self._skip_layer_ratio = skip_layer_ratio
        self.__decoder = decoder.create(
            patch_embedding_scales=patch_embedding_scales,
            input_dims=image_dims,
            output_dims=(num_classes, height, width),
            dropout_rate=decoder_dropout_rate,
        )
        self.__encoder_dropout_rate = encoder_dropout_rate
        self.__patch_fusion_dropout_rate = patch_fusion_dropout_rate
        self.__num_encoder_heads = num_encoder_heads

    @property
    def image_dims(self) -> _t.Tuple[int]:
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
    def decoder(self) -> _decoder.BaseDecoder:
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
    def apply_encoder_stage(
            self,
            patch_embeddings: _t.Dict[str, _torch.Tensor],
            return_attention_weights: bool = False,
    ) -> _t.Union[
        _t.Dict[str, _torch.Tensor],
        _t.Tuple[
            _t.Dict[str, _torch.Tensor],
            _t.Dict[str, _t.List[_torch.Tensor]]
        ]
    ]:
        """
        Apply the encoder stage to the input tensors.

        :param patch_embeddings: The patch embeddings for each scale.
        :param return_attention_weights: Whether to return the attention weights.
        :return: Encoded tensors for each scale.
        """

    def forward(
            self, x: _torch.Tensor, return_attention_weights: bool = False
    ) -> _t.Union[
        _torch.Tensor,
        _t.Tuple[
            _torch.Tensor,
            _t.Dict[str, _t.List[_torch.Tensor]],
        ]
    ]:
        """
        Forward pass of the vision transformer model. This method applies the patch embedding, encoder, and decoder
        stages to the input tensor. Each stage returns a dictionary of tensors that are passed to the next stage. These
        represent the different patch embedding scales.

        :param x: The input tensor.
        :param return_attention_weights: Whether to return the attention weights.
        :return: The output tensor and optional attention weights.
        """
        image_dims = self.__image_dims
        decoder = self.__decoder

        # Patch Embedding
        patch_embeddings = self.apply_patch_embedding_stage(x)
        # Encoder Stage
        if return_attention_weights:
            patch_embeddings, attention_weights = self.apply_encoder_stage(
                patch_embeddings=patch_embeddings,
                return_attention_weights=return_attention_weights
            )
        else:
            patch_embeddings = self.apply_encoder_stage(
                patch_embeddings=patch_embeddings,
                return_attention_weights=return_attention_weights
            )
        # Decoder Stage
        x1 = decoder(patch_embeddings)
        assert x1.shape[2:] == image_dims

        if return_attention_weights:
            return x1, attention_weights

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
        encoder_dropout_rate = self.__encoder_dropout_rate
        num_encoder_heads = self.__num_encoder_heads

        kwargs = {
            'nhead': num_encoder_heads,
            'dropout': encoder_dropout_rate,
            'activation': _f.gelu,
            'batch_first': True,
        }
        encoders_scale_X = _nn.ModuleList(
            [
                _transformer_encoder_layer.TransformerEncoderLayer(
                    d_model=embed_dim,
                    dim_feedforward=int(embed_dim * 2),
                    **kwargs,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        return encoders_scale_X

    def _create_swin_encoder_layers_for_scale_X(
            self, *, embed_dim: int, input_resolution: _t.Tuple[int, int]
    ) -> '_nn.ModuleList[_swin_transformer_encoder.SwinTransformerBlock]':
        """
        Create Swin Transformer encoder layers for scale X.

        :param embed_dim: Patch embedding dimension.
        :param input_resolution: The resolution of the input tensor.
        :return: Module list of Swin Transformer encoder layers.
        """
        num_encoder_layers = self.__num_encoder_layers
        encoder_dropout_rate = self.__encoder_dropout_rate
        num_encoder_heads = self.__num_encoder_heads

        H = input_resolution[0]
        window_size = max(H // 4, 4)  # 4 is the minimum window size, with 4 patches in each window
        shift_size = window_size // 2
        shift_size = [shift_size, shift_size]
        kwargs = {
            'num_heads': num_encoder_heads,
            'dropout': encoder_dropout_rate,
            'attention_dropout': encoder_dropout_rate,
            'mlp_ratio': 2.0,
            'window_size': [window_size, window_size],
        }
        encoders_scale_X = _nn.ModuleList(
            [
                _swin_transformer_encoder.SwinTransformerBlock(
                    dim=embed_dim,
                    shift_size=[0, 0] if (i % 2 == 0) else shift_size,
                    **kwargs,
                )
                for i in range(num_encoder_layers)
            ]
        )

        return encoders_scale_X

    def _create_patch_fusion_layers_for_scale_X(
            self, *, in_dims: _t.List[_t.List[int]], out_patches: int, out_embed: int
    ) -> '_nn.ModuleList[_patch_fusion.PatchFusion]':
        """
        Create patch fusion layers for scale X to all other scales.

        :param in_dims: The dimensions of the input tensors.
        :param out_patches: Out number of patches.
        :param out_embed: Out patch embedding dimension.
        :return: Module list of patch fusion layers.
        """
        num_patch_fusion_layers = self.__num_patch_fusion_layers
        patch_fusion_dropout_rate = self.__patch_fusion_dropout_rate

        kwargs = {
            'in_dims': in_dims,
            'out_patches': out_patches,
            'out_embed': out_embed,
            'dropout_rate': patch_fusion_dropout_rate,
        }
        patch_fusion_layers_scale_X = _nn.ModuleList(
            [
                _patch_fusion.PatchFusion(
                    **kwargs,
                )
                for _ in range(num_patch_fusion_layers)
            ]
        )

        return patch_fusion_layers_scale_X
