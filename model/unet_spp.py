import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetSPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetSPP, self).__init__()

        # Encoder
        self.encoder1 = self.contracting_block(in_channels, 64)
        self.encoder2 = self.contracting_block(64, 128)
        self.encoder3 = self.contracting_block(128, 256)
        self.encoder4 = self.contracting_block(256, 512)

        # SPP
        self.spp = SpatialPyramidPooling(512, 64)

        # Decoder
        self.decoder1 = self.expanding_block(512 + 64, 256)
        self.decoder2 = self.expanding_block(512, 128)
        self.decoder3 = self.expanding_block(256, 64)
        self.decoder4 = self.expanding_block(128, 64)

        # Output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def contracting_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return block

    def expanding_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 2, out_channels, kernel_size=2, stride=2)
        )
        return block

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        # SPP
        spp = self.spp(enc4)

        dec1 = self.decoder1(torch.cat([spp, enc4], dim=1))
        dec2 = self.decoder2(torch.cat([dec1, enc3], dim=1))
        dec3 = self.decoder3(torch.cat([dec2, enc2], dim=1))
        dec4 = self.decoder4(torch.cat([dec3, enc1], dim=1))

        output = self.final_conv(dec4)

        return output


class SpatialPyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialPyramidPooling, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Pooling levels
        pool1 = F.avg_pool2d(x, kernel_size=(height, width))
        pool2 = F.avg_pool2d(x, kernel_size=(height // 2, width // 2))
        pool3 = F.avg_pool2d(x, kernel_size=(height // 3, width // 3))
        pool4 = F.avg_pool2d(x, kernel_size=(height // 6, width // 6))

        # Resize the pooled feature maps
        pool1 = F.interpolate(pool1, size=(height, width), mode='bilinear', align_corners=False)
        pool2 = F.interpolate(pool2, size=(height, width), mode='bilinear', align_corners=False)
        pool3 = F.interpolate(pool3, size=(height, width), mode='bilinear', align_corners=False)
        pool4 = F.interpolate(pool4, size=(height, width), mode='bilinear', align_corners=False)

        # Concatenate the pooled feature maps
        spp = torch.cat([x, pool1, pool2, pool3, pool4], dim=1)
        spp = self.conv1x1(spp)

        return spp

if __name__=='__main__':
    # Example usage
    input_ch = 1
    out_channels = 3

    unet = UNetSPP(input_ch, out_channels)
    input_tensor = torch.randn(5, input_ch, 384, 384)  # Example input tensor
    output_tensor = unet(input_tensor)
    print(output_tensor.shape)