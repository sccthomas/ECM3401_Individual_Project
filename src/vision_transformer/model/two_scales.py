import typing as _t

import torch as _torch

import src.vision_transformer.common.patch_embedding as _patch_embedding
import src.vision_transformer.model.base as _base


class SemanticSegmentationVisionTransformer(_base.SemanticSegmentationVisionTransformerBase):
    """
    Semantic Segmentation Vision Transformer for 2 scales.
    """

    def __init__(
            self,
            *,
            image_dims: _t.Tuple[int, int, int],
            num_encoder_layers: int,
            decoder_type: str,
            skip_layer_ratio: int,
            encoder_dropout_rate: float,
            patch_fusion_dropout_rate: float,
            decoder_dropout_rate: float,
            num_encoder_heads: int,
            num_classes: int,
            patch_embedding_scale_1: _t.Tuple[int, int],
            patch_embedding_scale_2: _t.Tuple[int, int],
    ) -> None:
        """
        Initialize the vision_transformer.

        :param image_dims: The dimensions of the input image.
        :param num_encoder_layers: The number of encoder layers
        :param decoder_type: The type of decoder to use.
        :param skip_layer_ratio: The ratio of encoder layers to skip for patch fusion.
        :param encoder_dropout_rate: The dropout rate in the encoder stage.
        :param patch_fusion_dropout_rate: The dropout rate in the patch fusion stage.
        :param decoder_dropout_rate: The dropout rate in the decoder stage.
        :param num_encoder_heads: The number of encoder heads.
        :param num_classes: The number of classes.
        :param patch_embedding_scale_1: The patch embedding configuration for scale 1.
        :param patch_embedding_scale_2: The patch embedding configuration for scale 2.
        """
        # Patch Embedding
        in_channels, height, width = image_dims

        super(SemanticSegmentationVisionTransformer, self).__init__(
            image_dims=image_dims,
            num_encoder_layers=num_encoder_layers,
            decoder_type=decoder_type,
            skip_layer_ratio=skip_layer_ratio,
            patch_embedding_scales=[patch_embedding_scale_1, patch_embedding_scale_2],
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

        # Encoder Stage
        # - Transformers Encoder Layers
        #  - Scale 1
        self.__encoders_scale_1 = self._create_encoder_layers_for_scale_X(
            embed_dim=patch_embedding_scale_1[1],
        )
        #  - Scale 2
        self.__encoders_scale_2 = self._create_encoder_layers_for_scale_X(
            embed_dim=patch_embedding_scale_2[1],
        )

        # - Patch Fusion Layers
        #   - Scale 1
        self.__patch_fusions_scale_1 = self._create_patch_fusion_layers_for_scale_X(
            in_dims=[[self.__patch_embedding_scale_2.num_patches, patch_embedding_scale_2[1]], ],
            out_patches=self.__patch_embedding_scale_1.num_patches,
            out_embed=patch_embedding_scale_1[1],
        )
        #   - Scale 2
        self.__patch_fusions_scale_2 = self._create_patch_fusion_layers_for_scale_X(
            in_dims=[[self.__patch_embedding_scale_1.num_patches, patch_embedding_scale_1[1]], ],
            out_patches=self.__patch_embedding_scale_2.num_patches,
            out_embed=patch_embedding_scale_2[1],
        )

    def apply_patch_embedding_stage(self, x: _torch.Tensor) -> _t.Dict[str, _torch.Tensor]:
        """
        Apply the patch embedding to the input tensor.

        :param x: The input tensor.
        :return: The patch embeddings for scale 1 and scale 2.
        """
        patch_embedding_scale_1 = self.__patch_embedding_scale_1
        patch_embedding_scale_2 = self.__patch_embedding_scale_2

        x1 = patch_embedding_scale_1(x)
        x2 = patch_embedding_scale_2(x)

        return {'x1': x1, 'x2': x2}

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

        :param patch_embeddings: The patch embeddings for scale 1 and scale 2.
        :param return_attention_weights: Whether to return the attention weights.
        :return: Encoded tensors for scale 1 and scale 2 and attention weights.
        """
        encoders_scale_1 = self.__encoders_scale_1
        encoders_scale_2 = self.__encoders_scale_2
        patch_fusions_scale_1 = self.__patch_fusions_scale_1
        patch_fusions_scale_2 = self.__patch_fusions_scale_2
        skip_layer_ratio = self._skip_layer_ratio

        # Encoder Stage
        x1 = patch_embeddings['x1']
        x2 = patch_embeddings['x2']

        if return_attention_weights:
            kwargs = {'return_attention_weights': return_attention_weights}
            weights = {
                'x1': [],
                'x2': [],
            }
            x1, weights_x1 = encoders_scale_1[0](x1, **kwargs)
            x2, weights_x2 = encoders_scale_2[0](x2, **kwargs)
            weights['x1'].append(weights_x1)
            weights['x2'].append(weights_x2)
        else:
            x1 = encoders_scale_1[0](x1)
            x2 = encoders_scale_2[0](x2)

        skip_layer = 0
        for layer, (encoder_scale_1, encoder_scale_2) in enumerate(
                zip(encoders_scale_1[1:], encoders_scale_2[1:]), start=1
        ):
            # - Patch Fusion Layer
            if layer % skip_layer_ratio == 0:
                x1 = patch_fusions_scale_1[skip_layer](target_tensor=x1, tensors=[x2])
                x2 = patch_fusions_scale_2[skip_layer](target_tensor=x2, tensors=[x1])
                # - Increment the skip layer
                skip_layer += 1

            # - Transformer Encoder Layer
            if return_attention_weights:
                x1, weights_x1 = encoder_scale_1(x1, **kwargs)
                x2, weights_x2 = encoder_scale_2(x2, **kwargs)
                weights['x1'].append(weights_x1)
                weights['x2'].append(weights_x2)
            else:
                x1 = encoder_scale_1(x1)
                x2 = encoder_scale_2(x2)

        if return_attention_weights:
            return {'x1': x1, 'x2': x2}, weights

        return {'x1': x1, 'x2': x2}
