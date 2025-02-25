import typing as _t

import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _f

import src.vision_transformer.common.decoder as _decoder
import src.vision_transformer.common.patch_embedding as _patch_embedding
import src.vision_transformer.common.patch_fusion as _patch_fusion
import src.vision_transformer.common.swin_transformer_encoder_layer as _swin_transformer_encoder
import src.vision_transformer.common.transformer_encoder_layer as _transformer_encoder_layer


class SemanticSegmentationVisionTransformer(_nn.Module):
    """
    Semantic Segmentation Vision Transformer for any number of scales.
    """

    def __init__(
            self,
            *,
            image_dims: _t.Tuple[int, int, int],
            num_encoder_layers: int,
            use_swin_transformer: bool,
            use_heavyweight_decoder: bool,
            skip_layer_ratio: int,
            use_learnable_skip_layers: bool,
            use_skip_layer_gated_attention: bool,
            encoder_dropout_rate: float,
            patch_fusion_dropout_rate: float,
            decoder_dropout_rate: float,
            num_encoder_heads: int,
            num_classes: int,
            patch_embedding_scale_1: _t.Tuple[int, int],
            patch_embedding_scale_2: _t.Tuple[int, int] = None,
            patch_embedding_scale_3: _t.Tuple[int, int] = None,
            patch_embedding_scale_4: _t.Tuple[int, int] = None,
            patch_embedding_scale_5: _t.Tuple[int, int] = None,
            window_size: str = "medium",
    ) -> None:
        """
        Initialize the vision_transformer.

        :param image_dims: The dimensions of the input image.
        :param num_encoder_layers: The number of encoder layers.
        :param use_swin_transformer: Whether to use the Swin Transformer encoder layer.
        :param use_heavyweight_decoder: Whether to use the heavyweight decoder.
        :param skip_layer_ratio: The ratio of encoder layers to skip for patch fusion.
        :param use_learnable_skip_layers: Whether to use learnable skip layers.
        :param use_skip_layer_gated_attention: Whether to use the skip layer gated attention.
        :param encoder_dropout_rate: The dropout rate in the encoder stage.
        :param patch_fusion_dropout_rate: The dropout rate in the patch fusion stage.
        :param decoder_dropout_rate: The dropout rate in the decoder stage.
        :param num_encoder_heads: The number of encoder heads.
        :param num_classes: The number of classes.
        :param patch_embedding_scale_1: The patch embedding configuration for scale 1.
        :param patch_embedding_scale_2: Optional, The patch embedding configuration for scale 2.
        :param patch_embedding_scale_3: Optional, The patch embedding configuration for scale 3.
        :param patch_embedding_scale_4: Optional, The patch embedding configuration for scale 4.
        :param patch_embedding_scale_5: Optional, The patch embedding configuration for scale 5.
        :param window_size: The window size for the Swin Transformer encoder.
        """
        in_channels, height, width = image_dims

        assert height == width, "Input image must be square."

        super(SemanticSegmentationVisionTransformer, self).__init__()

        # Model Hyperparameters
        self.__image_dims = image_dims[1:]
        self.__num_encoder_layers = num_encoder_layers
        self.__num_patch_fusion_layers = (num_encoder_layers // skip_layer_ratio) - 1
        self.__skip_layer_ratio = skip_layer_ratio
        self.__encoder_dropout_rate = encoder_dropout_rate
        self.__patch_fusion_dropout_rate = patch_fusion_dropout_rate
        self.__num_encoder_heads = num_encoder_heads
        self.__use_skip_layer_gated_attention = use_skip_layer_gated_attention
        self.__use_learnable_skip_layers = use_learnable_skip_layers
        self.__window_size = 8 if window_size == "small" else 4 if window_size == "medium" else 2

        # Create Patch Embedding Modules
        # - Determine which patch embedding scales are present
        patch_embedding_scales = [
            patch_embedding_scale
            for patch_embedding_scale in
            [
                patch_embedding_scale_1,
                patch_embedding_scale_2,
                patch_embedding_scale_3,
                patch_embedding_scale_4,
                patch_embedding_scale_5
            ]
            if patch_embedding_scale is not None
        ]
        # - Create the patch embedding modules
        kwargs = {'in_channels': in_channels, 'image_size': height}
        patch_embedding_modules = _nn.ModuleDict(
            {
                f"x{i + 1}": _patch_embedding.PatchEmbedding(
                    patch_size=patch_embedding_scale[0],
                    embed_dim=patch_embedding_scale[1],
                    **kwargs,
                )
                for i, patch_embedding_scale in enumerate(patch_embedding_scales)
            }
        )
        self.__patch_embedding_modules = patch_embedding_modules

        # Encoder Stage
        # - Transformers Encoder Layers
        encoders = _nn.ModuleDict()
        create_encoder_layer = (
            self.__create_swin_encoder_layers_for_scale_X
            if use_swin_transformer else
            self.__create_encoder_layers_for_scale_X
        )
        for key, patch_embedding_module in patch_embedding_modules.items():
            encoders[key] = create_encoder_layer(
                patch_embedding=patch_embedding_module
            )
        self.__encoders = encoders

        # - Patch Fusion Layers
        patch_fusions = _nn.ModuleDict()
        for key, patch_embedding_module in patch_embedding_modules.items():
            patch_fusions[key] = self.__create_patch_fusion_layers_for_scale_X(
                in_patch_embeddings=[
                    patch_embedding_modules[other_key]
                    for other_key in patch_embedding_modules.keys()
                    if other_key != key
                ],
                out_patch_embedding=patch_embedding_module,
            )
        self.__patch_fusions = patch_fusions

        # Decoder stage
        # - Determine what type of decoder to use
        if use_heavyweight_decoder:
            decoder = _decoder.HeavyWeightDecoder
        else:
            decoder = _decoder.LightWeightDecoder

        self.__decoder = decoder.create(
            patch_embedding_scales=patch_embedding_scales,
            input_dims=image_dims,
            output_dims=(num_classes, height, width),
            dropout_rate=decoder_dropout_rate,
        )

    @property
    def encoder(self) -> _nn.ModuleDict:
        """
        Get the encoder modules.

        :return: The encoder modules.
        """
        return self.__encoders

    @property
    def decoder(self) -> _decoder.BaseDecoder:
        """
        Get the decoder module.

        :return: The decoder module.
        """
        return self.__decoder

    def forward(
            self, x: _torch.Tensor, keep_attention_scores: bool = False,
    ) -> _torch.Tensor:
        """
        Forward pass of the vision transformer model. This method applies the patch embedding, encoder, and decoder
        stages to the input tensor. Each stage returns a dictionary of tensors that are passed to the next stage. These
        represent the different patch embedding scales.

        :param x: The input tensor.
        :param keep_attention_scores: Whether to store the attention scores.
        :return: The output tensor and optional attention weights.
        """
        image_dims = self.__image_dims
        decoder = self.__decoder

        # Patch Embedding
        patch_embeddings = self.apply_patch_embedding_stage(x)

        # Encoder Stage
        patch_embeddings = self.apply_encoder_stage(
            patch_embeddings=patch_embeddings, keep_attention_scores=keep_attention_scores
        )

        # Decoder Stage
        x1 = decoder(patch_embeddings)

        # Assert that the correct dimensions have been reached
        assert x1.shape[2:] == image_dims

        return x1

    def apply_patch_embedding_stage(self, x: _torch.Tensor) -> _t.Dict[str, _torch.Tensor]:
        """
        Apply the patch embedding to the input tensor.

        :param x: The input tensor.
        :return: The patch embeddings for all available scales.
        """
        patch_embedding_modules = self.__patch_embedding_modules

        patch_embeddings = {
            key: patch_embedding_module(x)
            for key, patch_embedding_module in patch_embedding_modules.items()
        }

        return patch_embeddings

    def apply_encoder_stage(
            self,
            patch_embeddings: _t.Dict[str, _torch.Tensor],
            keep_attention_scores: bool = False,
    ) -> _t.Dict[str, _torch.Tensor]:
        """
        Apply the encoder stage to the input tensors.

        :param patch_embeddings: The patch embeddings for all available scales.
        :param keep_attention_scores: Whether to store the attention scores.
        :return: The encoded tensors for all available scales.
        """
        encoders = self.__encoders
        patch_fusions = self.__patch_fusions
        skip_layer_ratio = self.__skip_layer_ratio
        num_encoders = self.__num_encoder_layers

        # Encoder Stage
        for layer in range(0, num_encoders):
            # - Patch Fusion Layer
            if layer % skip_layer_ratio == 0 and layer > 0:
                skip_layer = layer // skip_layer_ratio - 1
                for key, patch_embedding in patch_embeddings.items():
                    patch_embeddings[key] = patch_fusions[key][skip_layer](
                        target_tensor=patch_embedding,
                        tensors=[
                            other_embedding for other_key, other_embedding in patch_embeddings.items()
                            if key != other_key
                        ],
                        keep_attention_scores=keep_attention_scores,
                    )

            # - Encoder Layer
            for key in patch_embeddings:
                patch_embeddings[key] = encoders[key][layer](
                    patch_embeddings[key], keep_attention_scores=keep_attention_scores
                )

        return patch_embeddings

    def get_attention_scores(
            self
    ) -> _t.Dict[_t.Tuple[str, int], _t.Dict[str, _t.List[_torch.Tensor]]]:
        """
        Get the attention scores for all encoder layers for a given tensor x.

        :return: The attention scores for all encoder layers.
        """
        encoders = self.__encoders
        patch_fusions = self.__patch_fusions
        patch_embedding_modules = self.__patch_embedding_modules

        # Get Attention Scores
        attention_scores = {
            key: {}
            for key in patch_embedding_modules.keys()
        }
        for key in encoders:
            attention_scores[key]['encoder'] = [
                attn_scores
                for encoder in encoders[key].children()
                if (attn_scores := encoder.attention_scores) is not None
            ]
        for key in patch_fusions:
            attention_scores[key]['patch_fusion'] = [
                attn_scores
                for patch_fusion in patch_fusions[key].children()
                if (attn_scores := patch_fusion.attention_scores) is not None
            ]

        # Add Patch Size Information.
        attention_scores = {
            (key, patch_embedding_module.patch_size): attention_scores[key]
            for key, patch_embedding_module in patch_embedding_modules.items()
        }

        return attention_scores

    def __create_encoder_layers_for_scale_X(
            self, patch_embedding: _patch_embedding.PatchEmbedding
    ) -> '_nn.ModuleList[_nn.TransformerEncoderLayer]':
        """
        Create Classic Transformer encoder layers for scale X.

        :param patch_embedding: The patch embedding layer for scale X.
        :return: Module list of Transformer encoder layers.
        """
        num_encoder_layers = self.__num_encoder_layers
        encoder_dropout_rate = self.__encoder_dropout_rate
        num_encoder_heads = self.__num_encoder_heads

        embed_dim = patch_embedding.embed_dim
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

    def __create_swin_encoder_layers_for_scale_X(
            self, patch_embedding: _patch_embedding.PatchEmbedding
    ) -> '_nn.ModuleList[_swin_transformer_encoder.SwinTransformerBlock]':
        """
        Create Swin Transformer encoder layers for scale X.

        :param patch_embedding: The patch embedding layer for scale X.
        :return: Module list of Swin Transformer encoder layers.
        """
        num_encoder_layers = self.__num_encoder_layers
        encoder_dropout_rate = self.__encoder_dropout_rate
        num_encoder_heads = self.__num_encoder_heads
        window_size = self.__window_size

        embed_dim = patch_embedding.embed_dim
        H = patch_embedding.H

        window_size = max(H // window_size, 2)  # H / number_of_windows, but at least 4 windows
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

    def __create_patch_fusion_layers_for_scale_X(
            self,
            *,
            in_patch_embeddings: _t.List[_patch_embedding.PatchEmbedding],
            out_patch_embedding: _patch_embedding.PatchEmbedding
    ) -> '_nn.ModuleList[_patch_fusion.PatchFusionLearnable]':
        """
        Create patch fusion layers for scale X to all other scales.

        :param in_patch_embeddings: The input patch embeddings for scale X.
        :param out_patch_embedding: The output patch embedding for scale X
        :return: Module list of patch fusion layers.
        """
        num_patch_fusion_layers = self.__num_patch_fusion_layers
        patch_fusion_dropout_rate = self.__patch_fusion_dropout_rate
        use_skip_layer_gated_attention = self.__use_skip_layer_gated_attention
        use_learnable_skip_layers = self.__use_learnable_skip_layers

        in_dims = [
            [patch_embedding.num_patches, patch_embedding.embed_dim]
            for patch_embedding in in_patch_embeddings
        ]
        out_patches = out_patch_embedding.num_patches
        out_embed = out_patch_embedding.embed_dim
        kwargs = {
            'in_dims': in_dims,
            'out_patches': out_patches,
            'out_embed': out_embed,
            'dropout_rate': patch_fusion_dropout_rate,
            'use_gated_attention': use_skip_layer_gated_attention,
            'num_heads': 2 if use_skip_layer_gated_attention else None,
        }
        patch_fusion_module = (
            _patch_fusion.PatchFusionLearnable
            if use_learnable_skip_layers else
            _patch_fusion.PatchFusionNonLearnable
        )
        patch_fusion_layers_scale_X = _nn.ModuleList(
            [
                patch_fusion_module(
                    **kwargs,
                )
                for _ in range(num_patch_fusion_layers)
            ]
        )

        return patch_fusion_layers_scale_X
