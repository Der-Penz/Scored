import torch
from torch import nn


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1, g: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, k // 2, groups=g, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = ConvBNAct(in_ch, in_ch, k=3, s=stride, g=in_ch)
        self.pw = ConvBNAct(in_ch, out_ch, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))
