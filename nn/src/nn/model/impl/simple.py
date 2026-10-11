import timm
import torch
import torch.nn.functional as F
from nn.model.impl.util_modules import ConvBNAct, DepthwiseSeparableConv
from nn.model.parts import YoloBackbone, YoloHead, YoloNeck
from torch import nn

# Assuming your abstract base classes (YoloBackbone, YoloNeck, YoloHead, YoloParts) are imported


class TimmYoloBackbone(YoloBackbone):
    """
    YOLO backbone powered by `timm` pretrained models.

    Parameters
    ----------
    model_name : str
        The name of the timm backbone (e.g., 'mobilenetv4_conv_small.e1200_r224_in1k',
        'lcnet_100', 'fastvit_t8', 'ghostnet_100').
    pretrained : bool
        Whether to load pretrained ImageNet weights.
    out_indices : tuple[int, ...]
        The indices of the feature maps to extract. Default (2, 3, 4) corresponds
        to strides 8, 16, and 32 (P3, P4, P5) in most standard architectures.
    """

    def __init__(
        self,
        model_name: str = "mobilenetv4_conv_small.e1200_r224_in1k",
        pretrained: bool = True,
        in_channels: int = 3,
        out_indices: tuple[int, ...] = (2, 3, 4),
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )

        self._channels = tuple(self.backbone.feature_info.channels())
        self._strides = tuple(self.backbone.feature_info.reduction())

        if in_channels == 3:
            first_conv: nn.Conv2d = self.backbone.conv_stem
            new_conv = nn.Conv2d(
                in_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )
            self.backbone.conv_stem = new_conv
        else:
            raise NotImplementedError(
                "Currently, only in_channels=3 or in_channels=1 is supported for TimmYoloBackbone."
            )

    def get_backbone_channels(self) -> tuple[int, ...]:
        return self._channels

    def get_strides(self) -> tuple[int, ...]:
        return self._strides

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.backbone(x)


class MobileYoloNeck(YoloNeck):
    """PANet Neck adapting dynamically to any channel sizes from timm backbones."""

    def __init__(self, channels: tuple[int, ...]):
        super().__init__(channels)
        c1, c2, c3 = channels

        self.lat_p5 = ConvBNAct(c3, c2, k=1)
        self.fpn_p4 = DepthwiseSeparableConv(c2 + c2, c2)

        self.lat_p4 = ConvBNAct(c2, c1, k=1)
        self.fpn_p3 = DepthwiseSeparableConv(c1 + c1, c1)

        # Bottom-up path
        self.down_p3 = DepthwiseSeparableConv(c1, c1, stride=2)
        self.pan_p4 = DepthwiseSeparableConv(c1 + c2, c2)

        self.down_p4 = DepthwiseSeparableConv(c2, c2, stride=2)
        self.pan_p5 = DepthwiseSeparableConv(c2 + c3, c3)

        self._neck_channels = (c1, c2, c3)

    def get_neck_channels(self) -> tuple[int, ...]:
        return self._neck_channels

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        p3, p4, p5 = x

        # FPN Top-Down
        p5_up = F.interpolate(self.lat_p5(p5), size=p4.shape[2:], mode="nearest")
        n4 = self.fpn_p4(torch.cat([p4, p5_up], dim=1))

        n4_up = F.interpolate(self.lat_p4(n4), size=p3.shape[2:], mode="nearest")
        out_p3 = self.fpn_p3(torch.cat([p3, n4_up], dim=1))

        # PAN Bottom-Up
        p3_down = self.down_p3(out_p3)
        out_p4 = self.pan_p4(torch.cat([n4, p3_down], dim=1))

        p4_down = self.down_p4(out_p4)
        out_p5 = self.pan_p5(torch.cat([p5, p4_down], dim=1))

        return [out_p3, out_p4, out_p5]


class MobileYoloHead(YoloHead):
    """Decoupled head producing classification, box regression, and keypoints."""

    def __init__(self, n_classes: int, num_kpts: int, channels: tuple[int, ...], reg_k: int = 16):
        super().__init__(n_classes, num_kpts, channels)

        self.cls_heads = nn.ModuleList()
        self.reg_heads = nn.ModuleList()
        self.kpt_heads = nn.ModuleList()
        self.reg_k = reg_k

        assert num_kpts >= 0, "Number of keypoints must be non-negative and above zero."

        kpt_ch = num_kpts * 3  # (x, y, visibility)

        for c in channels:
            self.cls_heads.append(
                nn.Sequential(
                    DepthwiseSeparableConv(c, c),
                    nn.Conv2d(c, n_classes, kernel_size=1),
                )
            )
            self.reg_heads.append(
                nn.Sequential(
                    DepthwiseSeparableConv(c, c),
                    nn.Conv2d(c, 4 * reg_k, kernel_size=1),
                )
            )
            self.kpt_heads.append(
                nn.Sequential(
                    DepthwiseSeparableConv(c, c),
                    nn.Conv2d(c, kpt_ch, kernel_size=1),
                )
            )

    def forward(
        self, x: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        cls_preds, reg_preds, kpt_preds = [], [], []

        for feat, cls_h, reg_h, kpt_h in zip(
            x, self.cls_heads, self.reg_heads, self.kpt_heads
        ):
            cls_preds.append(cls_h(feat))
            reg_preds.append(reg_h(feat))
            kpt_preds.append(kpt_h(feat))

        return cls_preds, reg_preds, kpt_preds
