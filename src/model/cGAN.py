import torch
import torch.nn as nn

class GeneratorUNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super().__init__()
        
        def down(in_c, out_c, bn=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if bn: layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, True))
            return nn.Sequential(*layers)

        def up(in_c, out_c, dropout=False):
            layers = [
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(True)
            ]
            if dropout: layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        self.down1 = down(input_channels, 64, bn=False)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.down5 = down(512, 512)
        self.down6 = down(512, 512)

        self.up1 = up(512, 512)
        self.up2 = up(1024, 512)
        self.up3 = up(1024, 256)
        self.up4 = up(512, 128)
        self.up5 = up(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, output_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        bottleneck = self.down6(d5)

        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d5], 1))
        u3 = self.up3(torch.cat([u2, d4], 1))
        u4 = self.up4(torch.cat([u3, d3], 1))
        u5 = self.up5(torch.cat([u4, d2], 1))

        return self.final(torch.cat([u5, d1], 1))

class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels+1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, 4, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, img, label):
        x = torch.cat([img, label], dim=1)
        return self.model(x)