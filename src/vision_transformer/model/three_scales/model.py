import typing as _t

import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _f

import src.vision_transformer.common.decoder as _decoder
import src.vision_transformer.common.patch_embedding as _patch_embedding
import src.vision_transformer.common.patch_fusion as _patch_fusion


class SemanticSegmentationVisionTransformer(_nn.Module):
    """
    Semantic Segmentation Vision Transformer vision_transformer.
    """

    def __init__(
            self,
            image_dims: _t.Tuple[int, int, int],
            patch_embedding_scale_1: _t.Tuple[int, int],
            patch_embedding_scale_2: _t.Tuple[int, int],
            patch_embedding_scale_3: _t.Tuple[int, int],
    ) -> None:
        """
        Initialize the vision_transformer.

        :param image_dims: The dimensions of the input image.
        :param patch_embedding_scale_1: The patch embedding configuration for scale 1.
        :param patch_embedding_scale_2: The patch embedding configuration for scale 2.
        :param patch_embedding_scale_3: The patch embedding configuration for scale 3.
        """
        super(SemanticSegmentationVisionTransformer, self).__init__()

        in_channels, height, width = image_dims

        assert height == width, "Input image must be square."

        # Patch Embedding
        self.__patch_embedding_scale_1 = _patch_embedding.PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_embedding_scale_1[0],
            embed_dim=patch_embedding_scale_1[1],
            image_size=height
        )
        self.__patch_embedding_scale_2 = _patch_embedding.PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_embedding_scale_2[0],
            embed_dim=patch_embedding_scale_2[1],
            image_size=height
        )
        self.__patch_embedding_scale_3 = _patch_embedding.PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_embedding_scale_3[0],
            embed_dim=patch_embedding_scale_3[1],
            image_size=height
        )

        # Encoder Stage
        encoder_layers = 12
        self.__encoders_scale_1 = _nn.ModuleList(
            [
                _nn.TransformerEncoderLayer(
                    d_model=patch_embedding_scale_1[1],
                    nhead=16,
                    dim_feedforward=int(patch_embedding_scale_1[1] * 2),
                    dropout=0.1,
                    activation=_f.gelu,
                )
                for __ in range(encoder_layers)
            ]
        )
        self.__encoders_scale_2 = _nn.ModuleList(
            [
                _nn.TransformerEncoderLayer(
                    d_model=patch_embedding_scale_2[1],
                    nhead=16,
                    dim_feedforward=int(patch_embedding_scale_2[1] * 2),
                    dropout=0.1,
                    activation=_f.gelu,
                )
                for __ in range(encoder_layers)
            ]
        )
        self.__encoders_scale_3 = _nn.ModuleList(
            [
                _nn.TransformerEncoderLayer(
                    d_model=patch_embedding_scale_3[1],
                    nhead=16,
                    dim_feedforward=int(patch_embedding_scale_3[1] * 2),
                    dropout=0.1,
                    activation=_f.gelu,
                )
                for __ in range(encoder_layers)
            ]
        )
        patch_fusion_layers = encoder_layers - 1
        # - Scale 1
        self.__patch_fusions_scale_1_to_2 = _nn.ModuleList(
            [
                _patch_fusion.PatchFusion(
                    in_patches=self.__patch_embedding_scale_1.num_patches,
                    in_embed=patch_embedding_scale_1[1],
                    out_patches=self.__patch_embedding_scale_2.num_patches,
                    out_embed=patch_embedding_scale_2[1]
                )
                for __ in range(patch_fusion_layers)
            ]
        )
        self.__patch_fusions_scale_1_to_3 = _nn.ModuleList(
            [
                _patch_fusion.PatchFusion(
                    in_patches=self.__patch_embedding_scale_1.num_patches,
                    in_embed=patch_embedding_scale_1[1],
                    out_patches=self.__patch_embedding_scale_3.num_patches,
                    out_embed=patch_embedding_scale_3[1]
                )
                for __ in range(patch_fusion_layers)
            ]
        )
        # - Scale 2
        self.__patch_fusions_scale_2_to_1 = _nn.ModuleList(
            [
                _patch_fusion.PatchFusion(
                    in_patches=self.__patch_embedding_scale_2.num_patches,
                    in_embed=patch_embedding_scale_2[1],
                    out_patches=self.__patch_embedding_scale_1.num_patches,
                    out_embed=patch_embedding_scale_1[1]
                )
                for __ in range(patch_fusion_layers)
            ]
        )
        self.__patch_fusions_scale_2_to_3 = _nn.ModuleList(
            [
                _patch_fusion.PatchFusion(
                    in_patches=self.__patch_embedding_scale_2.num_patches,
                    in_embed=patch_embedding_scale_2[1],
                    out_patches=self.__patch_embedding_scale_3.num_patches,
                    out_embed=patch_embedding_scale_3[1]
                )
                for __ in range(patch_fusion_layers)
            ]
        )
        # - Scale 3
        self.__patch_fusions_scale_3_to_1 = _nn.ModuleList(
            [
                _patch_fusion.PatchFusion(
                    in_patches=self.__patch_embedding_scale_3.num_patches,
                    in_embed=patch_embedding_scale_3[1],
                    out_patches=self.__patch_embedding_scale_1.num_patches,
                    out_embed=patch_embedding_scale_1[1]
                )
                for __ in range(patch_fusion_layers)
            ]
        )
        self.__patch_fusions_scale_3_to_2 = _nn.ModuleList(
            [
                _patch_fusion.PatchFusion(
                    in_patches=self.__patch_embedding_scale_3.num_patches,
                    in_embed=patch_embedding_scale_3[1],
                    out_patches=self.__patch_embedding_scale_2.num_patches,
                    out_embed=patch_embedding_scale_2[1]
                )
                for __ in range(patch_fusion_layers)
            ]
        )

        # Decoder Stage
        # - Basically this code here will always upsample a tensor of shape [B, 256, X] to [B, 1, 256, 256]
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

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        """
        Forward pass.

        :param x: The input tensor.
        :return: The output tensor.
        """
        # Patch Embedding
        x1 = self.__patch_embedding_scale_1(x)
        x2 = self.__patch_embedding_scale_2(x)
        x3 = self.__patch_embedding_scale_3(x)

        # Encoder Stage
        x1 = self.__encoders_scale_1[0](x1)
        x2 = self.__encoders_scale_2[0](x2)
        x3 = self.__encoders_scale_3[0](x3)

        for (
                # Encoder Scales
                encoder_scale_1,
                encoder_scale_2,
                encoder_scale_3,
                # Patch Fusion Scale 1
                patch_fusion_scale_1_to_2,
                patch_fusion_scale_1_to_3,
                # Patch Fusion Scale 2
                patch_fusion_scale_2_to_1,
                patch_fusion_scale_2_to_3,
                # Patch Fusion Scale 3
                patch_fusion_scale_3_to_1,
                patch_fusion_scale_3_to_2,
        ) in zip(
            self.__encoders_scale_1[1:],
            self.__encoders_scale_2[1:],
            self.__encoders_scale_3[1:],
            self.__patch_fusions_scale_1_to_2,
            self.__patch_fusions_scale_1_to_3,
            self.__patch_fusions_scale_2_to_1,
            self.__patch_fusions_scale_2_to_3,
            self.__patch_fusions_scale_3_to_1,
            self.__patch_fusions_scale_3_to_2,
        ):
            # - Patch Fusion Layer
            #   - Scale 1
            x1 = patch_fusion_scale_2_to_1(x2, x1)
            x1 = patch_fusion_scale_3_to_1(x3, x1)
            #   - Scale 2
            x2 = patch_fusion_scale_1_to_2(x1, x2)
            x2 = patch_fusion_scale_3_to_2(x3, x2)
            #   - Scale 3
            x3 = patch_fusion_scale_1_to_3(x1, x3)
            x3 = patch_fusion_scale_2_to_3(x2, x3)

            # - Transformer Encoder Layer
            x1 = encoder_scale_1(x1)
            x2 = encoder_scale_2(x2)
            x3 = encoder_scale_3(x3)

        # Decoder Stage
        # - Upsample x3 to x2, x2 to x1
        x2 = self.__decoder_patch_fusion_scale_3_to_2(x3, x2)
        x1 = self.__decoder_patch_fusion_scale_2_to_1(x2, x1)
        # - Final Decoder Head
        x1 = self.__decoder(x1)

        return x1
