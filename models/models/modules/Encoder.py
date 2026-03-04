import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 64):
        """
        自适应输入 / 输出通道的编码器。
        Args:
            in_channels: 输入特征的通道数（例如 1、3、4 等）
            out_channels: 最后输出特征的通道数
            base_channels: 第一层卷积的基础通道数，默认为 64

            结合你当前数据的使用方式示例
            你的 Dataset 中，最终输出的是：
            ir_Y, vis_Y, vis_Cb, vis_Cr  # 形状都是 (B, 1, 256, 256)
            有几种常见使用方法：
            把四个通道拼在一起编码：
            x = torch.cat([ir_Y, vis_Y, vis_Cb, vis_Cr], dim=1)   # (B, 4, 256, 256)
            encoder = Encoder(in_channels=4, out_channels=512)
            feat = encoder(x)   # (B, 512, 32, 32)

            只对 Y 通道编码（红外 + 可见 Y 拼一起）：
            x_y = torch.cat([ir_Y, vis_Y], dim=1)   # (B, 2, 256, 256)
            encoder_y = Encoder(in_channels=2,out_channels=512)
            feat_y = encoder_y(x_y)

            对单通道 Y 独立编码（例如红外 Y）：
            encoder_ir = Encoder(in_channels=1, out_channels=512)
            feat_ir = encoder_ir(ir_Y)

        """
        super(Encoder, self).__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(c3, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

if __name__ == "__main__":
    x = torch.randn(32, 1, 256, 256)
    encoder = Encoder(in_channels=1, out_channels=512)
    feat = encoder(x)
    print(feat.shape)