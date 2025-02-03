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
        #   - Scale 1
        self.__encoders_scale_1 = self._create_encoder_layers_for_scale_X(
            embed_dim=patch_embedding_scale_1[1],
        )
        #   - Scale 2
        self.__encoders_scale_2 = self._create_encoder_layers_for_scale_X(
            embed_dim=patch_embedding_scale_2[1],
        )
        #   - Scale 3
        self.__encoders_scale_3 = self._create_encoder_layers_for_scale_X(
            embed_dim=patch_embedding_scale_3[1],
        )
        #   - Scale 4
        self.__encoders_scale_4 = self._create_encoder_layers_for_scale_X(
            embed_dim=patch_embedding_scale_4[1],
        )
        #   - Scale 5
        self.__encoders_scale_5 = self._create_encoder_layers_for_scale_X(
            embed_dim=patch_embedding_scale_5[1],
        )

        # - Patch Fusion Layers
        #   - Scale 1
        kwargs = {'in_patches': self.__patch_embedding_scale_1.num_patches, 'in_embed': patch_embedding_scale_1[1]}
        self.__patch_fusions_scale_1_to_2 = self._create_patch_fusion_layers_for_scale_X_to_Y(
            out_patches=self.__patch_embedding_scale_2.num_patches,
            out_embed=patch_embedding_scale_2[1],
            **kwargs
        )
        self.__patch_fusions_scale_1_to_3 = self._create_patch_fusion_layers_for_scale_X_to_Y(
            out_patches=self.__patch_embedding_scale_3.num_patches,
            out_embed=patch_embedding_scale_3[1],
            **kwargs
        )
        self.__patch_fusions_scale_1_to_4 = self._create_patch_fusion_layers_for_scale_X_to_Y(
            out_patches=self.__patch_embedding_scale_4.num_patches,
            out_embed=patch_embedding_scale_4[1],
            **kwargs
        )
        self.__patch_fusions_scale_1_to_5 = self._create_patch_fusion_layers_for_scale_X_to_Y(
            out_patches=self.__patch_embedding_scale_5.num_patches,
            out_embed=patch_embedding_scale_5[1],
            **kwargs
        )
        #   - Scale 2
        kwargs = {'in_patches': self.__patch_embedding_scale_2.num_patches, 'in_embed': patch_embedding_scale_2[1]}
        self.__patch_fusions_scale_2_to_1 = self._create_patch_fusion_layers_for_scale_X_to_Y(
            out_patches=self.__patch_embedding_scale_1.num_patches,
            out_embed=patch_embedding_scale_1[1],
            **kwargs
        )
        self.__patch_fusions_scale_2_to_3 = self._create_patch_fusion_layers_for_scale_X_to_Y(
            out_patches=self.__patch_embedding_scale_3.num_patches,
            out_embed=patch_embedding_scale_3[1],
            **kwargs
        )
        self.__patch_fusions_scale_2_to_4 = self._create_patch_fusion_layers_for_scale_X_to_Y(
            out_patches=self.__patch_embedding_scale_4.num_patches,
            out_embed=patch_embedding_scale_4[1],
            **kwargs
        )
        self.__patch_fusions_scale_2_to_5 = self._create_patch_fusion_layers_for_scale_X_to_Y(
            out_patches=self.__patch_embedding_scale_5.num_patches,
            out_embed=patch_embedding_scale_5[1],
            **kwargs
        )
        #   - Scale 3
        kwargs = {'in_patches': self.__patch_embedding_scale_3.num_patches, 'in_embed': patch_embedding_scale_3[1]}
        self.__patch_fusions_scale_3_to_1 = self._create_patch_fusion_layers_for_scale_X_to_Y(
            out_patches=self.__patch_embedding_scale_1.num_patches,
            out_embed=patch_embedding_scale_1[1],
            **kwargs
        )
        self.__patch_fusions_scale_3_to_2 = self._create_patch_fusion_layers_for_scale_X_to_Y(
            out_patches=self.__patch_embedding_scale_2.num_patches,
            out_embed=patch_embedding_scale_2[1],
            **kwargs
        )
        self.__patch_fusions_scale_3_to_4 = self._create_patch_fusion_layers_for_scale_X_to_Y(
            out_patches=self.__patch_embedding_scale_4.num_patches,
            out_embed=patch_embedding_scale_4[1],
            **kwargs
        )
        self.__patch_fusions_scale_3_to_5 = self._create_patch_fusion_layers_for_scale_X_to_Y(
            out_patches=self.__patch_embedding_scale_5.num_patches,
            out_embed=patch_embedding_scale_5[1],
            **kwargs
        )
        #   - Scale 4
        kwargs = {'in_patches': self.__patch_embedding_scale_4.num_patches, 'in_embed': patch_embedding_scale_4[1]}
        self.__patch_fusions_scale_4_to_1 = self._create_patch_fusion_layers_for_scale_X_to_Y(
            out_patches=self.__patch_embedding_scale_1.num_patches,
            out_embed=patch_embedding_scale_1[1],
            **kwargs
        )
        self.__patch_fusions_scale_4_to_2 = self._create_patch_fusion_layers_for_scale_X_to_Y(
            out_patches=self.__patch_embedding_scale_2.num_patches,
            out_embed=patch_embedding_scale_2[1],
            **kwargs
        )
        self.__patch_fusions_scale_4_to_3 = self._create_patch_fusion_layers_for_scale_X_to_Y(
            out_patches=self.__patch_embedding_scale_3.num_patches,
            out_embed=patch_embedding_scale_3[1],
            **kwargs
        )
        self.__patch_fusions_scale_4_to_5 = self._create_patch_fusion_layers_for_scale_X_to_Y(
            out_patches=self.__patch_embedding_scale_5.num_patches,
            out_embed=patch_embedding_scale_5[1],
            **kwargs
        )
        #   - Scale 5
        kwargs = {'in_patches': self.__patch_embedding_scale_5.num_patches, 'in_embed': patch_embedding_scale_5[1]}
        self.__patch_fusions_scale_5_to_1 = self._create_patch_fusion_layers_for_scale_X_to_Y(
            out_patches=self.__patch_embedding_scale_1.num_patches,
            out_embed=patch_embedding_scale_1[1],
            **kwargs
        )
        self.__patch_fusions_scale_5_to_2 = self._create_patch_fusion_layers_for_scale_X_to_Y(
            out_patches=self.__patch_embedding_scale_2.num_patches,
            out_embed=patch_embedding_scale_2[1],
            **kwargs
        )
        self.__patch_fusions_scale_5_to_3 = self._create_patch_fusion_layers_for_scale_X_to_Y(
            out_patches=self.__patch_embedding_scale_3.num_patches,
            out_embed=patch_embedding_scale_3[1],
            **kwargs
        )
        self.__patch_fusions_scale_5_to_4 = self._create_patch_fusion_layers_for_scale_X_to_Y(
            out_patches=self.__patch_embedding_scale_4.num_patches,
            out_embed=patch_embedding_scale_4[1],
            **kwargs
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
    ) -> _t.Tuple[_t.Dict[str, _torch.Tensor], _t.Dict[str, _t.List[_t.Optional[_torch.Tensor]]]]:
        """
        Apply the encoder stage to the input tensors.

        :param patch_embeddings: The patch embeddings for the 5 scales.
        :param return_attention_weights: Whether to return the attention weights.
        :return: The output tensors for the 5 scales encoded.
        """
        encoders_scale_1 = self.__encoders_scale_1
        encoders_scale_2 = self.__encoders_scale_2
        encoders_scale_3 = self.__encoders_scale_3
        encoders_scale_4 = self.__encoders_scale_4
        encoders_scale_5 = self.__encoders_scale_5
        patch_fusions_scale_1_to_2 = self.__patch_fusions_scale_1_to_2
        patch_fusions_scale_1_to_3 = self.__patch_fusions_scale_1_to_3
        patch_fusions_scale_1_to_4 = self.__patch_fusions_scale_1_to_4
        patch_fusions_scale_1_to_5 = self.__patch_fusions_scale_1_to_5
        patch_fusions_scale_2_to_1 = self.__patch_fusions_scale_2_to_1
        patch_fusions_scale_2_to_3 = self.__patch_fusions_scale_2_to_3
        patch_fusions_scale_2_to_4 = self.__patch_fusions_scale_2_to_4
        patch_fusions_scale_2_to_5 = self.__patch_fusions_scale_2_to_5
        patch_fusions_scale_3_to_1 = self.__patch_fusions_scale_3_to_1
        patch_fusions_scale_3_to_2 = self.__patch_fusions_scale_3_to_2
        patch_fusions_scale_3_to_4 = self.__patch_fusions_scale_3_to_4
        patch_fusions_scale_3_to_5 = self.__patch_fusions_scale_3_to_5
        patch_fusions_scale_4_to_1 = self.__patch_fusions_scale_4_to_1
        patch_fusions_scale_4_to_2 = self.__patch_fusions_scale_4_to_2
        patch_fusions_scale_4_to_3 = self.__patch_fusions_scale_4_to_3
        patch_fusions_scale_4_to_5 = self.__patch_fusions_scale_4_to_5
        patch_fusions_scale_5_to_1 = self.__patch_fusions_scale_5_to_1
        patch_fusions_scale_5_to_2 = self.__patch_fusions_scale_5_to_2
        patch_fusions_scale_5_to_3 = self.__patch_fusions_scale_5_to_3
        patch_fusions_scale_5_to_4 = self.__patch_fusions_scale_5_to_4

        kwargs = {'return_attention_weights': return_attention_weights}
        weights = {
            'x1': [],
            'x2': [],
            'x3': [],
            'x4': [],
            'x5': [],
        }

        # Encoder Stage
        x1 = patch_embeddings['x1']
        x2 = patch_embeddings['x2']
        x3 = patch_embeddings['x3']
        x4 = patch_embeddings['x4']
        x5 = patch_embeddings['x5']
        x1, weights_x1 = encoders_scale_1[0](x1, **kwargs)
        x2, weights_x2 = encoders_scale_2[0](x2, **kwargs)
        x3, weights_x3 = encoders_scale_3[0](x3, **kwargs)
        x4, weights_x4 = encoders_scale_4[0](x4, **kwargs)
        x5, weights_x5 = encoders_scale_5[0](x5, **kwargs)

        # Append the weights
        if return_attention_weights:
            weights['x1'].append(weights_x1)
            weights['x2'].append(weights_x2)
            weights['x3'].append(weights_x3)
            weights['x4'].append(weights_x4)
            weights['x5'].append(weights_x5)

        for (
                # Encoder Scales
                encoder_scale_1, encoder_scale_2, encoder_scale_3, encoder_scale_4, encoder_scale_5,
                # Patch Fusion Scale 1
                patch_fusion_scale_1_to_2, patch_fusion_scale_1_to_3, patch_fusion_scale_1_to_4,
                patch_fusion_scale_1_to_5,
                # Patch Fusion Scale 2
                patch_fusion_scale_2_to_1, patch_fusion_scale_2_to_3, patch_fusion_scale_2_to_4,
                patch_fusion_scale_2_to_5,
                # Patch Fusion Scale 3
                patch_fusion_scale_3_to_1, patch_fusion_scale_3_to_2, patch_fusion_scale_3_to_4,
                patch_fusion_scale_3_to_5,
                # Patch Fusion Scale 4
                patch_fusion_scale_4_to_1, patch_fusion_scale_4_to_2, patch_fusion_scale_4_to_3,
                patch_fusion_scale_4_to_5,
                # Patch Fusion Scale 5
                patch_fusion_scale_5_to_1, patch_fusion_scale_5_to_2, patch_fusion_scale_5_to_3,
                patch_fusion_scale_5_to_4,
        ) in zip(
            # Encoder Scales
            encoders_scale_1[1:], encoders_scale_2[1:], encoders_scale_3[1:], encoders_scale_4[1:],
            encoders_scale_5[1:],
            # Patch Fusion Scale 1
            patch_fusions_scale_1_to_2, patch_fusions_scale_1_to_3, patch_fusions_scale_1_to_4,
            patch_fusions_scale_1_to_5,
            # Patch Fusion Scale 2
            patch_fusions_scale_2_to_1, patch_fusions_scale_2_to_3, patch_fusions_scale_2_to_4,
            patch_fusions_scale_2_to_5,
            # Patch Fusion Scale 3
            patch_fusions_scale_3_to_1, patch_fusions_scale_3_to_2, patch_fusions_scale_3_to_4,
            patch_fusions_scale_3_to_5,
            # Patch Fusion Scale 4
            patch_fusions_scale_4_to_1, patch_fusions_scale_4_to_2, patch_fusions_scale_4_to_3,
            patch_fusions_scale_4_to_5,
            # Patch Fusion Scale 5
            patch_fusions_scale_5_to_1, patch_fusions_scale_5_to_2, patch_fusions_scale_5_to_3,
            patch_fusions_scale_5_to_4,
        ):
            # - Patch Fusion Layer
            #   - Scale 1
            x1_fused = patch_fusion_scale_2_to_1(x2, x1)
            x1_fused = patch_fusion_scale_3_to_1(x3, x1_fused)
            x1_fused = patch_fusion_scale_4_to_1(x4, x1_fused)
            x1_fused = patch_fusion_scale_5_to_1(x5, x1_fused)
            #   - Scale 2
            x2_fused = patch_fusion_scale_1_to_2(x1, x2)
            x2_fused = patch_fusion_scale_3_to_2(x3, x2_fused)
            x2_fused = patch_fusion_scale_4_to_2(x4, x2_fused)
            x2_fused = patch_fusion_scale_5_to_2(x5, x2_fused)
            #   - Scale 3
            x3_fused = patch_fusion_scale_1_to_3(x1, x3)
            x3_fused = patch_fusion_scale_2_to_3(x2, x3_fused)
            x3_fused = patch_fusion_scale_4_to_3(x4, x3_fused)
            x3_fused = patch_fusion_scale_5_to_3(x5, x3_fused)
            #   - Scale 4
            x4_fused = patch_fusion_scale_1_to_4(x1, x4)
            x4_fused = patch_fusion_scale_2_to_4(x2, x4_fused)
            x4_fused = patch_fusion_scale_3_to_4(x3, x4_fused)
            x4_fused = patch_fusion_scale_5_to_4(x5, x4_fused)
            #   - Scale 5
            x5_fused = patch_fusion_scale_1_to_5(x1, x5)
            x5_fused = patch_fusion_scale_2_to_5(x2, x5_fused)
            x5_fused = patch_fusion_scale_3_to_5(x3, x5_fused)
            x5_fused = patch_fusion_scale_4_to_5(x4, x5_fused)

            # - Transformer Encoder Layer
            x1, weights_x1 = encoder_scale_1(x1_fused, **kwargs)
            x2, weights_x2 = encoder_scale_2(x2_fused, **kwargs)
            x3, weights_x3 = encoder_scale_3(x3_fused, **kwargs)
            x4, weights_x4 = encoder_scale_4(x4_fused, **kwargs)
            x5, weights_x5 = encoder_scale_5(x5_fused, **kwargs)

            # - Append the weights
            if return_attention_weights:
                weights['x1'].append(weights_x1)
                weights['x2'].append(weights_x2)
                weights['x3'].append(weights_x3)
                weights['x4'].append(weights_x4)
                weights['x5'].append(weights_x5)

        return {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5}, weights
