import typing as _t

import torch as _torch

import src.vision_transformer.common.patch_embedding as _patch_embedding
import src.vision_transformer.model.base as _base


class SemanticSegmentationVisionTransformer(_base.SemanticSegmentationVisionTransformerBase):
    """
    Semantic Segmentation Vision Transformer for 5 scales.
    """

    def __init__(
            self,
            image_dims: _t.Tuple[int, int, int],
            num_encoder_layers: int,
            use_swin_transformer: bool,
            use_heavyweight_decoder: bool,
            skip_layer_ratio: int,
            use_skip_layer_gated_attention: bool,
            encoder_dropout_rate: float,
            patch_fusion_dropout_rate: float,
            decoder_dropout_rate: float,
            num_encoder_heads: int,
            num_classes: int,
            patch_embedding_scale_1: _t.Tuple[int, int],
            patch_embedding_scale_2: _t.Tuple[int, int],
            patch_embedding_scale_3: _t.Tuple[int, int],
            patch_embedding_scale_4: _t.Tuple[int, int],
            patch_embedding_scale_5: _t.Tuple[int, int],
    ) -> None:
        """
        Initialize the vision_transformer.

        :param image_dims: The dimensions of the input image.
        :param num_encoder_layers: The number of encoder layers.
        :param use_swin_transformer: Whether to use the Swin Transformer encoder layer.
        :param use_heavyweight_decoder: Whether to use the heavyweight decoder.
        :param skip_layer_ratio: The ratio of encoder layers to skip for patch fusion.
        :param use_skip_layer_gated_attention: Whether to use the skip layer gated attention.
        :param encoder_dropout_rate: The dropout rate in the encoder stage.
        :param patch_fusion_dropout_rate: The dropout rate in the patch fusion stage.
        :param decoder_dropout_rate: The dropout rate in the decoder stage.
        :param num_encoder_heads: The number of encoder heads.
        :param num_classes: The number of classes.
        :param patch_embedding_scale_1: The patch embedding configuration for scale 1.
        :param patch_embedding_scale_2: The patch embedding configuration for scale 2.
        :param patch_embedding_scale_3: The patch embedding configuration for scale 3.
        :param patch_embedding_scale_4: The patch embedding configuration for scale 4.
        :param patch_embedding_scale_5: The patch embedding configuration for scale 5.
        """
        # Patch Embedding
        in_channels, height, width = image_dims

        super(SemanticSegmentationVisionTransformer, self).__init__(
            image_dims=image_dims,
            num_encoder_layers=num_encoder_layers,
            use_swin_transformer=use_swin_transformer,
            use_heavyweight_decoder=use_heavyweight_decoder,
            skip_layer_ratio=skip_layer_ratio,
            use_skip_layer_gated_attention=use_skip_layer_gated_attention,
            patch_embedding_scales=[
                patch_embedding_scale_1,
                patch_embedding_scale_2,
                patch_embedding_scale_3,
                patch_embedding_scale_4,
                patch_embedding_scale_5,
            ],
            encoder_dropout_rate=encoder_dropout_rate,
            patch_fusion_dropout_rate=patch_fusion_dropout_rate,
            decoder_dropout_rate=decoder_dropout_rate,
            num_encoder_heads=num_encoder_heads,
            num_classes=num_classes,
        )

        kwargs = {'in_channels': in_channels, 'image_size': height}
        self.__patch_embedding_scale_1 = _patch_embedding.PatchEmbedding(
            patch_size=patch_embedding_scale_1[0],
            embed_dim=patch_embedding_scale_1[1],
            **kwargs,
        )
        self.__patch_embedding_scale_2 = _patch_embedding.PatchEmbedding(
            patch_size=patch_embedding_scale_2[0],
            embed_dim=patch_embedding_scale_2[1],
            **kwargs,
        )
        self.__patch_embedding_scale_3 = _patch_embedding.PatchEmbedding(
            patch_size=patch_embedding_scale_3[0],
            embed_dim=patch_embedding_scale_3[1],
            **kwargs,
        )
        self.__patch_embedding_scale_4 = _patch_embedding.PatchEmbedding(
            patch_size=patch_embedding_scale_4[0],
            embed_dim=patch_embedding_scale_4[1],
            **kwargs,
        )
        self.__patch_embedding_scale_5 = _patch_embedding.PatchEmbedding(
            patch_size=patch_embedding_scale_5[0],
            embed_dim=patch_embedding_scale_5[1],
            **kwargs,
        )

        # Encoder Stage
        # - Transformers Encoder Layers
        #  - Scale 1
        self.__encoders_scale_1 = self._create_encoder_layer(
            patch_embedding=self.__patch_embedding_scale_1
        )
        #  - Scale 2
        self.__encoders_scale_2 = self._create_encoder_layer(
            patch_embedding=self.__patch_embedding_scale_2
        )
        # - Scale 3
        self.__encoders_scale_3 = self._create_encoder_layer(
            patch_embedding=self.__patch_embedding_scale_3
        )
        # - Scale 4
        self.__encoders_scale_4 = self._create_encoder_layer(
            patch_embedding=self.__patch_embedding_scale_4
        )
        # - Scale 5
        self.__encoders_scale_5 = self._create_encoder_layer(
            patch_embedding=self.__patch_embedding_scale_5
        )

        # - Patch Fusion Layers
        #   - Scale 1
        self.__patch_fusions_scale_1 = self._create_patch_fusion_layers_for_scale_X(
            in_dims=[
                [self.__patch_embedding_scale_2.num_patches, patch_embedding_scale_2[1]],
                [self.__patch_embedding_scale_3.num_patches, patch_embedding_scale_3[1]],
                [self.__patch_embedding_scale_4.num_patches, patch_embedding_scale_4[1]],
                [self.__patch_embedding_scale_5.num_patches, patch_embedding_scale_5[1]],
            ],
            out_patches=self.__patch_embedding_scale_1.num_patches,
            out_embed=patch_embedding_scale_1[1],
        )
        #   - Scale 2
        self.__patch_fusions_scale_2 = self._create_patch_fusion_layers_for_scale_X(
            in_dims=[
                [self.__patch_embedding_scale_1.num_patches, patch_embedding_scale_1[1]],
                [self.__patch_embedding_scale_3.num_patches, patch_embedding_scale_3[1]],
                [self.__patch_embedding_scale_4.num_patches, patch_embedding_scale_4[1]],
                [self.__patch_embedding_scale_5.num_patches, patch_embedding_scale_5[1]],
            ],
            out_patches=self.__patch_embedding_scale_2.num_patches,
            out_embed=patch_embedding_scale_2[1],
        )
        #   - Scale 3
        self.__patch_fusions_scale_3 = self._create_patch_fusion_layers_for_scale_X(
            in_dims=[
                [self.__patch_embedding_scale_1.num_patches, patch_embedding_scale_1[1]],
                [self.__patch_embedding_scale_2.num_patches, patch_embedding_scale_2[1]],
                [self.__patch_embedding_scale_4.num_patches, patch_embedding_scale_4[1]],
                [self.__patch_embedding_scale_5.num_patches, patch_embedding_scale_5[1]],
            ],
            out_patches=self.__patch_embedding_scale_3.num_patches,
            out_embed=patch_embedding_scale_3[1],
        )
        #   - Scale 4
        self.__patch_fusions_scale_4 = self._create_patch_fusion_layers_for_scale_X(
            in_dims=[
                [self.__patch_embedding_scale_1.num_patches, patch_embedding_scale_1[1]],
                [self.__patch_embedding_scale_2.num_patches, patch_embedding_scale_2[1]],
                [self.__patch_embedding_scale_3.num_patches, patch_embedding_scale_3[1]],
                [self.__patch_embedding_scale_5.num_patches, patch_embedding_scale_5[1]],
            ],
            out_patches=self.__patch_embedding_scale_4.num_patches,
            out_embed=patch_embedding_scale_4[1],
        )
        #   - Scale 5
        self.__patch_fusions_scale_5 = self._create_patch_fusion_layers_for_scale_X(
            in_dims=[
                [self.__patch_embedding_scale_1.num_patches, patch_embedding_scale_1[1]],
                [self.__patch_embedding_scale_2.num_patches, patch_embedding_scale_2[1]],
                [self.__patch_embedding_scale_3.num_patches, patch_embedding_scale_3[1]],
                [self.__patch_embedding_scale_4.num_patches, patch_embedding_scale_4[1]],
            ],
            out_patches=self.__patch_embedding_scale_5.num_patches,
            out_embed=patch_embedding_scale_5[1],
        )

    def apply_patch_embedding_stage(self, x: _torch.Tensor) -> _t.Dict[str, _torch.Tensor]:
        """
        Apply the patch embedding to the input tensor.

        :param x: The input tensor.
        :return: The patch embeddings for the 5 scales.
        """
        patch_embedding_scale_1 = self.__patch_embedding_scale_1
        patch_embedding_scale_2 = self.__patch_embedding_scale_2
        patch_embedding_scale_3 = self.__patch_embedding_scale_3
        patch_embedding_scale_4 = self.__patch_embedding_scale_4
        patch_embedding_scale_5 = self.__patch_embedding_scale_5

        x1 = patch_embedding_scale_1(x)
        x2 = patch_embedding_scale_2(x)
        x3 = patch_embedding_scale_3(x)
        x4 = patch_embedding_scale_4(x)
        x5 = patch_embedding_scale_5(x)

        return {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5}

    def apply_encoder_stage(
            self,
            patch_embeddings: _t.Dict[str, _torch.Tensor],
            return_attention_weights: bool = False,
    ) -> _t.Tuple[_t.Dict[str, _torch.Tensor], _t.Optional[_t.Dict[str, _t.List[_torch.Tensor]]]]:
        """
        Apply the encoder stage to the input tensors.

        :param patch_embeddings: The patch embeddings for the 5 scales.
        :param return_attention_weights: Whether to return the attention weights.
        :return: The output tensors for the 5 scales encoded and optional attention weights.
        """
        encoders_scale_1 = self.__encoders_scale_1
        encoders_scale_2 = self.__encoders_scale_2
        encoders_scale_3 = self.__encoders_scale_3
        encoders_scale_4 = self.__encoders_scale_4
        encoders_scale_5 = self.__encoders_scale_5
        patch_fusions_scale_1 = self.__patch_fusions_scale_1
        patch_fusions_scale_2 = self.__patch_fusions_scale_2
        patch_fusions_scale_3 = self.__patch_fusions_scale_3
        patch_fusions_scale_4 = self.__patch_fusions_scale_4
        patch_fusions_scale_5 = self.__patch_fusions_scale_5
        skip_layer_ratio = self._skip_layer_ratio

        # Encoder Stage
        x1 = patch_embeddings['x1']
        x2 = patch_embeddings['x2']
        x3 = patch_embeddings['x3']
        x4 = patch_embeddings['x4']
        x5 = patch_embeddings['x5']

        kwargs = {'return_attention_weights': return_attention_weights}
        x1, weights_x1 = encoders_scale_1[0](x1, **kwargs)
        x2, weights_x2 = encoders_scale_2[0](x2, **kwargs)
        x3, weights_x3 = encoders_scale_3[0](x3, **kwargs)
        x4, weights_x4 = encoders_scale_4[0](x4, **kwargs)
        x5, weights_x5 = encoders_scale_5[0](x5, **kwargs)

        if return_attention_weights:
            weights = {
                'x1': [],
                'x2': [],
                'x3': [],
                'x4': [],
                'x5': [],
            }
            weights['x1'].append(weights_x1)
            weights['x2'].append(weights_x2)
            weights['x3'].append(weights_x3)
            weights['x4'].append(weights_x4)
            weights['x5'].append(weights_x5)
        else:
            weights = None

        skip_layer = 0
        for layer, (encoder_scale_1, encoder_scale_2, encoder_scale_3, encoder_scale_4, encoder_scale_5) in enumerate(
                zip(encoders_scale_1[1:], encoders_scale_2[1:], encoders_scale_3[1:], encoders_scale_4[1:],
                    encoders_scale_5[1:])
                , start=1
        ):
            # - Patch Fusion Layer
            if layer % skip_layer_ratio == 0:
                x1 = patch_fusions_scale_1[skip_layer](target_tensor=x1, tensors=[x2, x3, x4, x5])
                x2 = patch_fusions_scale_2[skip_layer](target_tensor=x2, tensors=[x1, x3, x4, x5])
                x3 = patch_fusions_scale_3[skip_layer](target_tensor=x3, tensors=[x1, x2, x4, x5])
                x4 = patch_fusions_scale_4[skip_layer](target_tensor=x4, tensors=[x1, x2, x3, x5])
                x5 = patch_fusions_scale_5[skip_layer](target_tensor=x5, tensors=[x1, x2, x3, x4])
                # - Increment the skip layer
                skip_layer += 1

            # - Transformer Encoder Layer
            x1, weights_x1 = encoder_scale_1(x1, **kwargs)
            x2, weights_x2 = encoder_scale_2(x2, **kwargs)
            x3, weights_x3 = encoder_scale_3(x3, **kwargs)
            x4, weights_x4 = encoder_scale_4(x4, **kwargs)
            x5, weights_x5 = encoder_scale_5(x5, **kwargs)

            if return_attention_weights:
                weights['x1'].append(weights_x1)
                weights['x2'].append(weights_x2)
                weights['x3'].append(weights_x3)
                weights['x4'].append(weights_x4)
                weights['x5'].append(weights_x5)

        return {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5}, weights
