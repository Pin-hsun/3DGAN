import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.encoder1 = self.contracting_block(in_channels, 64)
        self.encoder2 = self.contracting_block(64, 128)
        self.encoder3 = self.contracting_block(128, 256)
        self.encoder4 = self.contracting_block(256, 512)

        self.decoder1 = self.expanding_block(512, 256)
        self.decoder2 = self.expanding_block(512, 128)
        self.decoder3 = self.expanding_block(256, 64)
        self.decoder4 = self.expanding_block(128, 64)

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
        encoder1 = self.encoder1(x)
        encoder2 = self.encoder2(encoder1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)

        decoder1 = self.decoder1(encoder4)
        decoder2 = self.decoder2(torch.cat([decoder1, encoder3], dim=1))
        decoder3 = self.decoder3(torch.cat([decoder2, encoder2], dim=1))
        decoder4 = self.decoder4(torch.cat([decoder3, encoder1], dim=1))

        output = self.final_conv(decoder4)

        return output

if __name__=='__main__':
    # Example usage
    input_ch = 1
    out_channels = 3

    unet = UNet(input_ch, out_channels)
    input_tensor = torch.randn(5, input_ch, 384, 384)  # Example input tensor
    output_tensor = unet(input_tensor)
    print(output_tensor.shape)