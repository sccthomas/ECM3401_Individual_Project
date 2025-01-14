import torch.nn as _nn


class PatchFusion(_nn.Module):
    def __init__(self, in_patches, in_embed, out_patches, out_embed):
        super(PatchFusion, self).__init__()
        self.__feature_projector = _nn.Linear(in_embed, out_embed)

        self.__sequence_expander = _nn.ConvTranspose1d(
            in_channels=in_patches, out_channels=out_patches, kernel_size=1
        )

        self.__norm = _nn.LayerNorm(out_embed, eps=1e-6)

    def forward(self, x, y):
        x = self.__feature_projector(x)
        x = self.__sequence_expander(x)

        x = x + y

        x = self.__norm(x).float()

        return x
