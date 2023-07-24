import torch

class DoubleConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.step = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, X):
        self.step(X)

class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__

        self.layer1 = DoubleConvBlock(1, 64)
        self.layer2 = DoubleConvBlock(64, 128)
        self.layer3 = DoubleConvBlock(128, 256)
        self.layer4 = DoubleConvBlock(256, 512)

        self.layer5 = DoubleConvBlock(512+256, 256)
        self.layer6 = DoubleConvBlock(256+128, 128)
        self.layer7 = DoubleConvBlock(128+64, 64)
        self.layer8 = torch.nn.Conv2d(64, 1, 1)

        self.maxpool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.layer1(x)
        x1 = self.maxpool(x1)

        x2 = self.layer2(x1)
        x2 = self.maxpool(x2)

        x3 = self.layer3(x2)
        x3 = self.maxpool(x3)

        x4 = self.layer4(x3)

        x5 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x4)
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.layer5(x5)

        x6 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.layer6(x6)

        x7 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.layer7(x7)

        return self.layer8(x7)

