from typing import Tuple
import torch as _torch
import torch.nn as _nn


class Unet(_nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.down_convolutions_1 = _DownSample(in_channels, 64)
        self.down_convolutions_2 = _DownSample(in_channels=64, out_channels=128)
        self.down_convolutions_3 = _DownSample(in_channels=128, out_channels=256)
        self.down_convolutions_4 = _DownSample(in_channels=256, out_channels=512)

        self.bottle_neck = _DoubleConv(in_channels=512, out_channels=1024)

        self.up_convolutions_1 = _UpSample(in_channels=1024, out_channels=512)
        self.up_convolutions_2 = _UpSample(in_channels=512, out_channels=256)
        self.up_convolutions_3 = _UpSample(in_channels=256, out_channels=128)
        self.up_convolutions_4 = _UpSample(in_channels=128, out_channels=64)

        self.out = _nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        down_1, pooled_1 = self.down_convolutions_1(x)
        down_2, pooled_2 = self.down_convolutions_2(pooled_1)
        down_3, pooled_3 = self.down_convolutions_3(pooled_2)
        down_4, pooled_4 = self.down_convolutions_4(pooled_3)

        bottle_neck = self.bottle_neck(pooled_4)

        up_1 = self.up_convolutions_1(bottle_neck, down_4)
        up_2 = self.up_convolutions_2(up_1, down_3)
        up_3 = self.up_convolutions_3(up_2, down_2)
        up_4 = self.up_convolutions_4(up_3, down_1)

        return self.out(up_4)


# ------------------------------
# Private Helpers
# ------------------------------

class _DoubleConv(_nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = _nn.Sequential(
            _nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _nn.BatchNorm2d(out_channels),
            _nn.ReLU(inplace=True),
            _nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _nn.BatchNorm2d(out_channels),
            _nn.ReLU(inplace=True),
            _nn.Dropout(p=0.5),
        )

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        return self.double_conv(x)


class _DownSample(_nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = _DoubleConv(in_channels, out_channels)
        self.max_pool = _nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: _torch.Tensor) -> Tuple[_torch.Tensor, _torch.Tensor]:
        down = self.double_conv(x)
        pooled = self.max_pool(down)

        return down, pooled


class _UpSample(_nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = _nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.double_conv = _DoubleConv(in_channels, out_channels)

    def forward(self, x1: _torch.Tensor, x2: _torch.Tensor) -> _torch.Tensor:
        x1 = self.up(x1)
        x = _torch.cat([x1, x2], 1)

        return self.double_conv(x)
