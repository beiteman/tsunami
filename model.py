import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.pool_conv(x)
    
class Up(nn.Module):
    def __init__(self, decoder_ch, skip_ch, out_ch):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            decoder_ch, decoder_ch // 2,
            kernel_size=2, stride=2
        )

        self.conv = DoubleConv(
            decoder_ch // 2 + skip_ch,
            out_ch
        )

    def forward(self, x_dec, x_skip):
        x_dec = self.up(x_dec)

        # spatial safety (odd sizes)
        if x_dec.shape[-2:] != x_skip.shape[-2:]:
            x_dec = F.interpolate(
                x_dec, size=x_skip.shape[-2:],
                mode="bilinear", align_corners=False
            )

        x = torch.cat([x_skip, x_dec], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        self.up1 = Up(512, 256, 256)
        self.up2 = Up(256, 128, 128)
        self.up3 = Up(128, 64, 64)

        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)     # 64
        x2 = self.down1(x1)  # 128
        x3 = self.down2(x2)  # 256
        x4 = self.down3(x3)  # 512

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        return self.outc(x)
