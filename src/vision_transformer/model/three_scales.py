import typing as _t

import torch as _torch

import src.vision_transformer.common.patch_embedding as _patch_embedding
import src.vision_transformer.model.base as _base


class SemanticSegmentationVisionTransformer(_base.SemanticSegmentationVisionTransformerBase):
    """
    Semantic Segmentation Vision Transformer for 3 scales.
    """

    def __init__(
            self,
            image_dims: _t.Tuple[int, int, int],
            num_encoder_layers: int,
            patch_embedding_scale_1: _t.Tuple[int, int],
            patch_embedding_scale_2: _t.Tuple[int, int],
            patch_embedding_scale_3: _t.Tuple[int, int],
    ) -> None:
        """
        Initialize the vision_transformer.

        :param image_dims: The dimensions of the input image.
        :param num_encoder_layers: The number of encoder layers
        :param patch_embedding_scale_1: The patch embedding configuration for scale 1.
        :param patch_embedding_scale_2: The patch embedding configuration for scale 2.
        :param patch_embedding_scale_3: The patch embedding configuration for scale 3.
        """
        # Patch Embedding
        in_channels, height, width = image_dims

        super(SemanticSegmentationVisionTransformer, self).__init__(
            image_dims=image_dims,
            num_encoder_layers=num_encoder_layers,
            patch_embedding_scales=[patch_embedding_scale_1, patch_embedding_scale_2, patch_embedding_scale_3],
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
        # - Scale 3
        self.__encoders_scale_3 = self._create_encoder_layers_for_scale_X(
            embed_dim=patch_embedding_scale_3[1],
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

    def apply_patch_embedding_stage(self, x: _torch.Tensor) -> _t.Dict[str, _torch.Tensor]:
        """
        Apply the patch embedding to the input tensor.

        :param x: The input tensor.
        :return: The patch embeddings for the 3 scales.
        """
        patch_embedding_scale_1 = self.__patch_embedding_scale_1
        patch_embedding_scale_2 = self.__patch_embedding_scale_2
        patch_embedding_scale_3 = self.__patch_embedding_scale_3

        x1 = patch_embedding_scale_1(x)
        x2 = patch_embedding_scale_2(x)
        x3 = patch_embedding_scale_3(x)

        return {'x1': x1, 'x2': x2, 'x3': x3}

    def apply_encoder_stage(
            self,
            patch_embeddings: _t.Dict[str, _torch.Tensor],
            return_attention_weights: bool = False,
    ) -> _t.Tuple[_t.Dict[str, _torch.Tensor], _t.Dict[str, _t.List[_t.Optional[_torch.Tensor]]]]:
        """
        Apply the encoder stage to the input tensors.

        :param patch_embeddings: The patch embeddings for the 3 scales.
        :param return_attention_weights: Whether to return the attention weights.
        :return: The output tensors for the 3 scales encoded.
        """
        encoders_scale_1 = self.__encoders_scale_1
        encoders_scale_2 = self.__encoders_scale_2
        encoders_scale_3 = self.__encoders_scale_3
        patch_fusions_scale_1_to_2 = self.__patch_fusions_scale_1_to_2
        patch_fusions_scale_1_to_3 = self.__patch_fusions_scale_1_to_3
        patch_fusions_scale_2_to_1 = self.__patch_fusions_scale_2_to_1
        patch_fusions_scale_2_to_3 = self.__patch_fusions_scale_2_to_3
        patch_fusions_scale_3_to_1 = self.__patch_fusions_scale_3_to_1
        patch_fusions_scale_3_to_2 = self.__patch_fusions_scale_3_to_2

        kwargs = {'return_attention_weights': return_attention_weights}
        weights = {
            'x1': [],
            'x2': [],
            'x3': [],
        }

        # Encoder Stage
        x1 = patch_embeddings['x1']
        x2 = patch_embeddings['x2']
        x3 = patch_embeddings['x3']
        x1, weights_x1 = encoders_scale_1[0](x1, **kwargs)
        x2, weights_x2 = encoders_scale_2[0](x2, **kwargs)
        x3, weights_x3 = encoders_scale_3[0](x3, **kwargs)

        # Append the weights
        if return_attention_weights:
            weights['x1'].append(weights_x1)
            weights['x2'].append(weights_x2)
            weights['x3'].append(weights_x3)

        for (
                # Encoder Scales
                encoder_scale_1, encoder_scale_2, encoder_scale_3,
                # Patch Fusion Scale 1
                patch_fusion_scale_1_to_2, patch_fusion_scale_1_to_3,
                # Patch Fusion Scale 2
                patch_fusion_scale_2_to_1, patch_fusion_scale_2_to_3,
                # Patch Fusion Scale 3
                patch_fusion_scale_3_to_1, patch_fusion_scale_3_to_2,
        ) in zip(
            # Encoder Scales
            encoders_scale_1[1:], encoders_scale_2[1:], encoders_scale_3[1:],
            # Patch Fusion Scale 1
            patch_fusions_scale_1_to_2, patch_fusions_scale_1_to_3,
            # Patch Fusion Scale 2
            patch_fusions_scale_2_to_1, patch_fusions_scale_2_to_3,
            # Patch Fusion Scale 3
            patch_fusions_scale_3_to_1, patch_fusions_scale_3_to_2,
        ):
            # - Patch Fusion Layer
            #   - Scale 1
            x1_fused = patch_fusion_scale_2_to_1(x2, x1)
            x1_fused = patch_fusion_scale_3_to_1(x3, x1_fused)
            #   - Scale 2
            x2_fused = patch_fusion_scale_1_to_2(x1, x2)
            x2_fused = patch_fusion_scale_3_to_2(x3, x2_fused)
            #   - Scale 3
            x3_fused = patch_fusion_scale_1_to_3(x1, x3)
            x3_fused = patch_fusion_scale_2_to_3(x2, x3_fused)

            # - Transformer Encoder Layer
            x1, weights_x1 = encoder_scale_1(x1_fused, **kwargs)
            x2, weights_x2 = encoder_scale_2(x2_fused, **kwargs)
            x3, weights_x3 = encoder_scale_3(x3_fused, **kwargs)

            # - Append the weights
            if return_attention_weights:
                weights['x1'].append(weights_x1)
                weights['x2'].append(weights_x2)
                weights['x3'].append(weights_x3)

        return {'x1': x1, 'x2': x2, 'x3': x3}, weights
