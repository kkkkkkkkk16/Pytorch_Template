import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 64):
        """
        Takes (B, in_channels, 32, 32) and decodes to (B, out_channels, 256, 256)
        """
        super(Decoder, self).__init__()

        c3 = base_channels * 4
        c2 = base_channels * 2
        c1 = base_channels

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, c3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(c3, c2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(c2, c1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(c1, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid() # to constrain to 0-1 range
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

if __name__ == "__main__":
    x = torch.randn(32, 512, 32, 32)
    decoder = Decoder(in_channels=512, out_channels=1)
    out = decoder(x)
    print(out.shape)
