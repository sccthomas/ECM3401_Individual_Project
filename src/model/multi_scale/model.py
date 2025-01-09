import torch as _torch
import torch.nn as _nn

import src.model.multi_scale.config as _config
import src.model.multi_scale.decoder as _decoder
import src.model.multi_scale.encoder as _encoder
import src.model.multi_scale.patch_embedding as _patch_embedding


class SemanticSegmentationVisionTransformer(_nn.Module):
    def __init__(
            self,
            patch_embedding_modules: '_nn.ModuleList[_patch_embedding.PatchEmbedding]',
            encoder_module: _encoder.Encoder,
            decoder_module: _decoder.Decoder,
    ) -> None:
        """

        :param patch_embedding_modules: Patch embedding modules.
        :param encoder_module: Encoder module.
        :param decoder_module: Decoder module.
        """
        super(SemanticSegmentationVisionTransformer, self).__init__()
        self.__patch_embedding_modules = patch_embedding_modules
        self.__encoder_module = encoder_module
        self.__decoder_module = decoder_module

    @classmethod
    def from_config(cls, config: _config.ModelConfig) -> 'SemanticSegmentationVisionTransformer':
        """
        Create a Vision Transformer model from a configuration.

        :param config: The model configuration.
        :return: The Vision Transformer model.
        """
        C, H, _ = config.input_dimensions
        patch_embedding_modules = _nn.ModuleList(
            [
                _patch_embedding.PatchEmbedding(
                    in_channels=C,
                    out_channels=patch_embedding_config.in_channels,
                    patch_size=patch_embedding_config.patch_size,
                    image_size=H,
                )
                for patch_embedding_config in config.patch_embedding_configs
            ]
        )
        encoder_module = _encoder.Encoder.from_config(config.encoder_config)
        decoder_module = _decoder.Decoder.from_config(config.decoder_config)

        return cls(patch_embedding_modules, encoder_module, decoder_module)

    @property
    def patch_embedding_modules(self) -> '_nn.ModuleList[_patch_embedding.PatchEmbedding]':
        """
        :return: The patch embedding modules.
        """
        return self.__patch_embedding_modules

    @property
    def encoder_module(self) -> '_encoder.Encoder':
        """
        :return: The encoder module.
        """
        return self.__encoder_module

    @property
    def decoder_module(self) -> '_decoder.Decoder':
        """
        :return: The decoder module.
        """
        return self.__decoder_module

    def forward(self, x: '_torch.Tensor') -> '_torch.Tensor':
        """
        Forward pass of the model.

        :param x: The input tensor.
        :return: The output tensor.
        """
        patch_embedding_modules = self.__patch_embedding_modules
        encoder_module = self.__encoder_module
        decoder_module = self.__decoder_module

        # Create different patch embeddings for different scales.
        patch_embeddings = [
            patch_embedding_module(x).contiguous()
            for patch_embedding_module in patch_embedding_modules
        ]

        # Feed the patch embeddings to the encoder module.
        patch_embeddings = [
            patch_embedding.contiguous()
            for patch_embedding in encoder_module(patch_embeddings)
        ]

        # Feed the encoder output to the decoder module.
        segmentation_output = decoder_module(patch_embeddings).contiguous()

        # Return the final segmentation output.
        return segmentation_output
