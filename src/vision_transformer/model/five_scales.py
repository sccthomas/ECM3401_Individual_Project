import typing as _t

import torch as _torch

import src.vision_transformer.common.decoder as _decoder
import src.vision_transformer.common.patch_embedding as _patch_embedding
import src.vision_transformer.common.patch_fusion as _patch_fusion
import src.vision_transformer.model.base as _base


class SemanticSegmentationVisionTransformer(_base.SemanticSegmentationVisionTransformerBase):
    """
    Semantic Segmentation Vision Transformer for 5 scales.
    """

    def __init__(
            self,
            image_dims: _t.Tuple[int, int, int],
            num_encoder_layers: int,
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
        :param patch_embedding_scale_1: The patch embedding configuration for scale 1.
        :param patch_embedding_scale_2: The patch embedding configuration for scale 2.
        :param patch_embedding_scale_3: The patch embedding configuration for scale 3.
        :param patch_embedding_scale_4: The patch embedding configuration for scale 4.
        :param patch_embedding_scale_5: The patch embedding configuration for scale 5.
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
        kwargs = {
            'num_heads': 16,
            'drop': 0.25,
            'attn_drop': 0.25,
            'drop_path': 0.25,
        }
        #  - Scale 1
        self.__encoders_scale_1 = self._create_swin_encoder_layers_for_scale_X(
            H=self.__patch_embedding_scale_1.H,
            embed_dim=patch_embedding_scale_1[1],
            input_resolution=self.__patch_embedding_scale_1.resolution,
            kwargs=kwargs,
        )
        #  - Scale 2
        self.__encoders_scale_2 = self._create_swin_encoder_layers_for_scale_X(
            H=self.__patch_embedding_scale_2.H,
            embed_dim=patch_embedding_scale_2[1],
            input_resolution=self.__patch_embedding_scale_2.resolution,
            kwargs=kwargs,
        )
        # - Scale 3
        self.__encoders_scale_3 = self._create_swin_encoder_layers_for_scale_X(
            H=self.__patch_embedding_scale_3.H,
            embed_dim=patch_embedding_scale_3[1],
            input_resolution=self.__patch_embedding_scale_3.resolution,
            kwargs=kwargs,
        )
        self.__encoders_scale_4 = self._create_swin_encoder_layers_for_scale_X(
            H=self.__patch_embedding_scale_4.H,
            embed_dim=patch_embedding_scale_4[1],
            input_resolution=self.__patch_embedding_scale_4.resolution,
            kwargs=kwargs,
        )
        self.__encoders_scale_5 = self._create_swin_encoder_layers_for_scale_X(
            H=self.__patch_embedding_scale_5.H,
            embed_dim=patch_embedding_scale_5[1],
            input_resolution=self.__patch_embedding_scale_5.resolution,
            kwargs=kwargs,
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

        # Decoder Stage
        self.__decoder_patch_fusion_scale_5_to_4 = _patch_fusion.PatchFusion(
            in_patches=self.__patch_embedding_scale_5.num_patches,
            in_embed=patch_embedding_scale_5[1],
            out_patches=self.__patch_embedding_scale_4.num_patches,
            out_embed=patch_embedding_scale_4[1]
        )
        self.__decoder_patch_fusion_scale_4_to_3 = _patch_fusion.PatchFusion(
            in_patches=self.__patch_embedding_scale_4.num_patches,
            in_embed=patch_embedding_scale_4[1],
            out_patches=self.__patch_embedding_scale_3.num_patches,
            out_embed=patch_embedding_scale_3[1]
        )
        self.__decoder_patch_fusion_scale_3_to_2 = _patch_fusion.PatchFusion(
            in_patches=self.__patch_embedding_scale_3.num_patches,
            in_embed=patch_embedding_scale_3[1],
            out_patches=self.__patch_embedding_scale_2.num_patches,
            out_embed=patch_embedding_scale_2[1]
        )
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
            self, x1: _torch.Tensor, x2: _torch.Tensor, x3: _torch.Tensor, x4: _torch.Tensor, x5: _torch.Tensor,
    ) -> _t.Dict[str, _torch.Tensor]:
        """
        Apply the encoder stage to the input tensors.

        :param x1: Scale 1 input tensor.
        :param x2: Scale 2 input tensor.
        :param x3: Scale 3 input tensor.
        :param x4: Scale 4 input tensor.
        :param x5: Scale 5 input tensor.
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

        # Encoder Stage
        x1 = encoders_scale_1[0](x1)
        x2 = encoders_scale_2[0](x2)
        x3 = encoders_scale_3[0](x3)
        x4 = encoders_scale_4[0](x4)
        x5 = encoders_scale_5[0](x5)

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
            x1 = patch_fusion_scale_2_to_1(x2, x1)
            x1 = patch_fusion_scale_3_to_1(x3, x1)
            x1 = patch_fusion_scale_4_to_1(x4, x1)
            x1 = patch_fusion_scale_5_to_1(x5, x1)
            #   - Scale 2
            x2 = patch_fusion_scale_1_to_2(x1, x2)
            x2 = patch_fusion_scale_3_to_2(x3, x2)
            x2 = patch_fusion_scale_4_to_2(x4, x2)
            x2 = patch_fusion_scale_5_to_2(x5, x2)
            #   - Scale 3
            x3 = patch_fusion_scale_1_to_3(x1, x3)
            x3 = patch_fusion_scale_2_to_3(x2, x3)
            x3 = patch_fusion_scale_4_to_3(x4, x3)
            x3 = patch_fusion_scale_5_to_3(x5, x3)
            #   - Scale 4
            x4 = patch_fusion_scale_1_to_4(x1, x4)
            x4 = patch_fusion_scale_2_to_4(x2, x4)
            x4 = patch_fusion_scale_3_to_4(x3, x4)
            x4 = patch_fusion_scale_5_to_4(x5, x4)
            #   - Scale 5
            x5 = patch_fusion_scale_1_to_5(x1, x5)
            x5 = patch_fusion_scale_2_to_5(x2, x5)
            x5 = patch_fusion_scale_3_to_5(x3, x5)
            x5 = patch_fusion_scale_4_to_5(x4, x5)

            # - Transformer Encoder Layer
            x1 = encoder_scale_1(x1)
            x2 = encoder_scale_2(x2)
            x3 = encoder_scale_3(x3)
            x4 = encoder_scale_4(x4)
            x5 = encoder_scale_5(x5)

        return {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5}

    def apply_decoder_stage(
            self, x1: _torch.Tensor, x2: _torch.Tensor, x3: _torch.Tensor, x4: _torch.Tensor, x5: _torch.Tensor,
    ) -> _torch.Tensor:
        """
        Apply the decoder stage to the input tensors.

        :param x1: Scale 1 input tensor.
        :param x2: Scale 2 input tensor.
        :param x3: Scale 3 input tensor.
        :param x4: Scale 4 input tensor.
        :param x5: Scale 5 input tensor.
        :return: The output tensor.
        """
        decoder_patch_fusion_scale_5_to_4 = self.__decoder_patch_fusion_scale_5_to_4
        decoder_patch_fusion_scale_4_to_3 = self.__decoder_patch_fusion_scale_4_to_3
        decoder_patch_fusion_scale_3_to_2 = self.__decoder_patch_fusion_scale_3_to_2
        decoder_patch_fusion_scale_2_to_1 = self.__decoder_patch_fusion_scale_2_to_1
        decoder = self.__decoder

        # Upsample x5 to x4 ,x4 to x3, x3 to x2, x2 to x1
        x4 = decoder_patch_fusion_scale_5_to_4(x5, x4)
        x3 = decoder_patch_fusion_scale_4_to_3(x4, x3)
        x2 = decoder_patch_fusion_scale_3_to_2(x3, x2)
        x1 = decoder_patch_fusion_scale_2_to_1(x2, x1)

        # Final Decoder Head
        x1 = decoder(x1)

        return x1
