import torch
import torch.nn as nn
import torch.nn.init as init

def conv_relu_x2(in_channels, out_channels, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, n_classes, input_channels=3):
        super().__init__()
        self.downscaling1 = conv_relu_x2(input_channels, 64)
        self.downscaling2 = conv_relu_x2(64, 128)
        self.downscaling3 = conv_relu_x2(128, 256)
        self.downscaling4 = conv_relu_x2(256, 512)
        self.downscaling5 = conv_relu_x2(512, 1024)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.upscaling4 = conv_relu_x2(512 + 1024, 512)
        self.upscaling3 = conv_relu_x2(256 + 512, 256)
        self.upscaling2 = conv_relu_x2(128 + 256, 128)
        self.upscaling1 = conv_relu_x2(128 + 64, 64)

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):

        # Downscaling path
        conv_s1 = self.downscaling1(x)
        x = self.maxpool(conv_s1)

        conv_s2 = self.downscaling2(x)
        x = self.maxpool(conv_s2)

        conv_s3 = self.downscaling3(x)
        x = self.maxpool(conv_s3)

        conv_s4 = self.downscaling4(x)
        x = self.maxpool(conv_s4)

        conv_s5 = self.downscaling5(x)

        # Upscaling path
        x = self.upsample(conv_s5)
        x = torch.cat([x, conv_s4], dim=1)

        x = self.upscaling4(x)
        x = self.upsample(x)
        x = torch.cat([x, conv_s3], dim=1)

        x = self.upscaling3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv_s2], dim=1)

        x = self.upscaling2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv_s1], dim=1)

        x = self.upscaling1(x)

        output = self.final_conv(x)
        # output = torch.sigmoid(output)

        return output
    
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)