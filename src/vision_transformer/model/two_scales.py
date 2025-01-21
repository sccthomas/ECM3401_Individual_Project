import typing as _t

import torch as _torch

import src.vision_transformer.common.decoder as _decoder
import src.vision_transformer.common.patch_embedding as _patch_embedding
import src.vision_transformer.common.patch_fusion as _patch_fusion
import src.vision_transformer.model.base as _base


class SemanticSegmentationVisionTransformer(_base.SemanticSegmentationVisionTransformerBase):
    """
    Semantic Segmentation Vision Transformer for 2 scales.
    """

    def __init__(
            self,
            image_dims: _t.Tuple[int, int, int],
            num_encoder_layers: int,
            patch_embedding_scale_1: _t.Tuple[int, int],
            patch_embedding_scale_2: _t.Tuple[int, int],
    ) -> None:
        """
        Initialize the vision_transformer.

        :param image_dims: The dimensions of the input image.
        :param num_encoder_layers: The number of encoder layers
        :param patch_embedding_scale_1: The patch embedding configuration for scale 1.
        :param patch_embedding_scale_2: The patch embedding configuration for scale 2.
        """
        super(SemanticSegmentationVisionTransformer, self).__init__(
            image_dims=image_dims,
            num_encoder_layers=num_encoder_layers
        )

        in_channels, height, width = image_dims

        # Patch Embedding
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
        self.__patch_fusions_scale_1_to_2 = self._create_patch_fusion_layers_for_scale_X_to_Y(
            in_patches=self.__patch_embedding_scale_1.num_patches,
            in_embed=patch_embedding_scale_1[1],
            out_patches=self.__patch_embedding_scale_2.num_patches,
            out_embed=patch_embedding_scale_2[1],
        )
        #   - Scale 2
        self.__patch_fusions_scale_2_to_1 = self._create_patch_fusion_layers_for_scale_X_to_Y(
            in_patches=self.__patch_embedding_scale_2.num_patches,
            in_embed=patch_embedding_scale_2[1],
            out_patches=self.__patch_embedding_scale_1.num_patches,
            out_embed=patch_embedding_scale_1[1],
        )

        # Decoder Stage
        self.__decoder_patch_fusion_scale_2_to_1 = _patch_fusion.PatchFusion(
            in_patches=self.__patch_embedding_scale_2.num_patches,
            in_embed=patch_embedding_scale_2[1],
            out_patches=self.__patch_embedding_scale_1.num_patches,
            out_embed=patch_embedding_scale_1[1]
        )
        self.__decoder = _decoder.Decoder.create(
            final_num_patches=self.__patch_embedding_scale_1.num_patches,
            final_embed_dim=patch_embedding_scale_1[1],
            output_dims=(1, height, width)
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

    def apply_encoder_stage(self, x1: _torch.Tensor, x2: _torch.Tensor) -> _t.Dict[str, _torch.Tensor]:
        """
        Apply the encoder stage to the input tensors.

        :param x1: Scale 1 input tensor.
        :param x2: Scale 2 input tensor.
        :return: Encoded tensors for scale 1 and scale 2.
        """
        encoder_scale_1 = self.__encoders_scale_1
        encoder_scale_2 = self.__encoders_scale_2
        patch_fusions_scale_1_to_2 = self.__patch_fusions_scale_1_to_2
        patch_fusions_scale_2_to_1 = self.__patch_fusions_scale_2_to_1

        # Encoder Stage
        x1 = encoder_scale_1[0](x1)
        x2 = encoder_scale_2[0](x2)

        for encoder_scale_1, encoder_scale_2, patch_fusion_scale_1_to_2, patch_fusion_scale_2_to_1 in zip(
                encoder_scale_1[1:],
                encoder_scale_2[1:],
                patch_fusions_scale_1_to_2,
                patch_fusions_scale_2_to_1,
        ):
            # - Patch Fusion Layer
            x1_fused = patch_fusion_scale_2_to_1(x2, x1)
            x2_fused = patch_fusion_scale_1_to_2(x1, x2)

            # - Transformer Encoder Layer
            x1 = encoder_scale_1(x1_fused)
            x2 = encoder_scale_2(x2_fused)

        return {'x1': x1, 'x2': x2}

    def apply_decoder_stage(self, x1: _torch.Tensor, x2: _torch.Tensor) -> _torch.Tensor:
        """
        Apply the decoder stage to the input tensors.

        :param x1: Scale 1 input tensor.
        :param x2: Scale 2 input tensor.
        :return: The decoded tensor.
        """
        decoder_patch_fusion_scale_2_to_1 = self.__decoder_patch_fusion_scale_2_to_1
        decoder = self.__decoder

        x1 = decoder_patch_fusion_scale_2_to_1(x2, x1)
        x1 = decoder(x1)

        return x1
