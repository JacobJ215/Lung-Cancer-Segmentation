import torch

class DoubleConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Define a sequence of two convolutional layers followed by ReLU activation.
        self.step = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.ReLU()
        )

    def forward(self, X):
        # Pass the input tensor through the sequence of layers.
        return self.step(X)

class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define the encoder part of the UNet with four DoubleConvBlock layers.
        self.layer1 = DoubleConvBlock(1, 64)
        self.layer2 = DoubleConvBlock(64, 128)
        self.layer3 = DoubleConvBlock(128, 256)
        self.layer4 = DoubleConvBlock(256, 512)

        # Define the decoder part of the UNet with three DoubleConvBlock layers.
        # The input channels for each layer are the concatenation of the output of the previous decoder layer and the corresponding encoder layer.
        self.layer5 = DoubleConvBlock(512 + 256, 256)
        self.layer6 = DoubleConvBlock(256 + 128, 128)
        self.layer7 = DoubleConvBlock(128 + 64, 64)

        # Final 1x1 convolutional layer to produce the output segmentation mask.
        self.layer8 = torch.nn.Conv2d(64, 1, 1)

        # Max pooling layer with kernel size 2 and stride 2 to downsample the feature maps.
        self.maxpool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder part of the UNet:
        x1 = self.layer1(x)
        x1_m = self.maxpool(x1)

        x2 = self.layer2(x1_m)
        x2_m = self.maxpool(x2)

        x3 = self.layer3(x2_m)
        x3_m = self.maxpool(x3)

        x4 = self.layer4(x3_m)

        # Decoder part of the UNet with upsampling using bilinear interpolation (Upsample) and concatenation of skip connections.
        x5 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x4)
        x5 = torch.cat([x5, x3], dim=1) # Skip-connection
        x5 = self.layer5(x5)

        x6 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x5)
        x6 = torch.cat([x6, x2], dim=1) # Skip-connection
        x6 = self.layer6(x6)

        x7 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x6)
        x7 = torch.cat([x7, x1], dim=1) # Skip-connection
        x7 = self.layer7(x7)

        # Final 1x1 convolutional layer to produce the output segmentation mask.
        return self.layer8(x7)
