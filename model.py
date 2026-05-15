import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

'''
- Model definition borrowed directly from https://github.com/tfp-photonics/neurop_invdes
- Removed batch normalization and padding to utilize just vanilla FNO
'''

class FNOModel(nn.Module):
    def __init__(self, modes, width, blocks):
        super().__init__()
        self.fno_blocks = self.get_fno_blocks(blocks, modes, width)
        self.conv_in = self.get_conv(2, width)
        self.conv_out = self.get_conv(width, 2)

    def get_fno_blocks(self, blocks, modes, width):
        fno_blocks = nn.Sequential(
            *[self.get_fno_block(modes, width) for _ in range(blocks)]
        )
        return fno_blocks

    def forward(self, x):
        x = self.conv_in(x)
        x = self.fno_blocks(x)
        x = self.conv_out(x)
        return x

class FNOModel2d(FNOModel):
    @staticmethod
    def get_fno_block(modes, width):
        return FNOBlock2d(modes, width)

    @staticmethod
    def get_conv(in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, 1, bias=True)

    @staticmethod
    def get_pad(padding):
        return nn.ConstantPad2d(padding, 0.0)

class FNOBlock(nn.Module):
    def __init__(self, modes, width):
        super().__init__()
        self.act = nn.GELU()
        self.fftconv = self.get_fft_conv(modes, width)
        self.conv = self.get_real_conv(width)

    def forward(self, x):
        x = self.fftconv(x) + self.conv(x)
        x = self.act(x)
        return x

class FNOBlock2d(FNOBlock):
    @staticmethod
    def get_fft_conv(modes, width):
        return FFTConv2d(width, width, modes)

    @staticmethod
    def get_real_conv(width):
        return nn.Conv2d(width, width, 1, bias=False)

    @staticmethod
    def get_batch_norm(width):
        return nn.BatchNorm2d(width)

class FFTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()

        self.out_channels = out_channels
        self.modes = modes

        self.w = nn.Parameter(
            torch.rand(in_channels, out_channels, 2 * modes, modes, 2)
            / (in_channels * out_channels)
        )

    @staticmethod
    def cmul2d(a, b):
        return torch.einsum("bixy,ioxy->boxy", a, b)

    def forward(self, x):
        xs = x.shape
        device = x.device

        x = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            xs[0],
            self.out_channels,
            xs[-2],
            xs[-1] // 2 + 1,
            dtype=torch.cfloat,
            device=device,
        )

        m = self.modes
        cx = xs[-2] // 2
        x = torch.fft.fftshift(x, -2)
        out_ft[..., cx - m : cx + m, :m] = self.cmul2d(
            x[..., cx - m : cx + m, :m],
            torch.view_as_complex(self.w),
        )
        out_ft = torch.fft.ifftshift(out_ft, -2)

        x = torch.fft.irfft2(out_ft, xs[-2:])

        return x
    

'''
- Model definition borrowed directly from https://github.com/milesial/Pytorch-UNet
- U-Net: Convolutional Networks for Biomedical Image Segmentation
followed the architecture of https://pubs.acs.org/doi/10.1021/acsphotonics.5c00104
'''

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)