import torch
from torch import nn
from models.modules.Decoder import Decoder
from models.modules.Encoder import Encoder


class FushionNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, feat_channels: int = 256):
        super(FushionNet, self).__init__()
        self.encoder = Encoder(in_channels=in_channels, out_channels=feat_channels)
        self.decoder = Decoder(in_channels=feat_channels, out_channels=out_channels)

    def forward(self, ir, vis):
        # Forward both modalities through the same encoder
        feat_ir = self.encoder(ir)
        feat_vis = self.encoder(vis)
        
        # Element-wise maximum fusion logic
        feat_fused = torch.max(feat_ir, feat_vis)
        
        # Decode the fused features
        out = self.decoder(feat_fused)
        return out