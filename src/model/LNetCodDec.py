import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x)


class LNet(nn.Module):
    def __init__(self, in_channels=3, base=32):
        super().__init__()

        # Encoder
        self.e1 = DepthwiseSeparableConv(in_channels, base)
        self.e2 = DepthwiseSeparableConv(base, base * 2)
        self.e3 = DepthwiseSeparableConv(base * 2, base * 4)

        # Downsample
        self.down1 = nn.Conv2d(base, base, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(base * 2, base * 2, 3, stride=2, padding=1)

        # Decoder
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)

        self.d2 = DepthwiseSeparableConv(base * 4, base * 2)
        self.d1 = DepthwiseSeparableConv(base * 2, base)

        # Output
        self.out_conv = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        # Encoder
        x1 = self.e1(x)
        x2 = self.e2(self.down1(x1))
        x3 = self.e3(self.down2(x2))

        # Decoder
        d2 = self.up2(x3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.d2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.d1(d1)

        out = self.out_conv(d1)
        return out
    
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)