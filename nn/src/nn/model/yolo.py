from typing import NamedTuple, TypedDict

import torch
import torch.nn.functional as F
from nn.box_util import XYXYBox
from nn.model.parts import YoloParts
from nn.util import make_anchors


class DecodedPredictions(TypedDict):
    boxes: XYXYBox  # (B, Total_Anchors, 4)
    scores: torch.Tensor  # (B, Total_Anchors, n_classes)
    keypoints: torch.Tensor  # (B, Total_Anchors, num_kpts, 3)


class TrainingPredictions(NamedTuple):
    one2one: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
    one2many: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]


class YoloV10(torch.nn.Module):
    def __init__(
        self,
        n_classes: int,
        num_kpts: int,
        reg_max: int,
        parts: YoloParts,
        include_aux_head: bool = True,
    ):
        super().__init__()
        (
            self.backbone,
            self.neck,
            self.head,
            self.aux_head,
        ) = parts.build_parts(n_classes, num_kpts, reg_max, include_aux_head=include_aux_head)

    def forward(self, x) -> TrainingPredictions | DecodedPredictions:
        assert isinstance(x, torch.Tensor), (
            f"Expected input to be a torch.Tensor, but got {type(x)}"
        )

        backbone_feature_maps = self.backbone(x)

        neck_features = self.neck(backbone_feature_maps)

        if self.aux_head is None:  # Inference-only model
            assert self.training is False, (
                "Model should be in eval mode during inference when no aux_head is provided."
            )
            cls, reg, kpts = self.head(neck_features)
            return self.decode_raw_predictions(cls, reg, kpts)

        # training mode with aux_head

        one2one = self.head([feature.detach() for feature in neck_features])

        if not self.training:
            cls, reg, kpts = one2one
            return self.decode_raw_predictions(cls, reg, kpts)

        one2many = self.aux_head(neck_features)
        return TrainingPredictions(one2one, one2many)

    def num_boxes_per_scale(self, img_size: tuple[int, int]) -> tuple[int, ...]:
        """
        Returns the number of boxes per scale.

        Parameters
        ----------
        img_size : tuple[int, int]
            The size of the input image (height, width).
        Returns
        -------
        tuple[int, ...]
            A tuple containing the number of boxes per scale.
        """
        strides = self.backbone.get_strides()

        return tuple(img_size[0] // s * img_size[1] // s for s in strides)

    def decode_raw_predictions(
        self,
        cls: list[torch.Tensor],
        regs: list[torch.Tensor],
        kpts: list[torch.Tensor],
    ) -> DecodedPredictions:
        """
        Decodes raw multi-scale head predictions into bounding boxes, class scores, and keypoints.

        Parameters
        ----------
        cls : list[torch.Tensor]
            Classification feature maps per scale. Shape: (B, n_classes, H_i, W_i)
        regs : list[torch.Tensor]
            Box regression feature maps per scale. Shape: (B, 4, H_i, W_i) or (B, 4 * reg_max, H_i, W_i)
        kpts : list[torch.Tensor]
            Keypoint feature maps per scale. Shape: (B, num_kpts * 3, H_i, W_i)

        Returns
        -------
        DecodedPredictions
            Dictionary containing:
            - "boxes": (B, Total_Anchors, 4) in (x1, y1, x2, y2) pixel space
            - "scores": (B, Total_Anchors, n_classes) class probabilities
            - "keypoints": (B, Total_Anchors, num_kpts, 3) in (x, y, visibility) pixel space
        """
        strides = self.backbone.get_strides()

        decoded_boxes = []
        decoded_scores = []
        decoded_keypoints = []

        for c, r, k, stride in zip(cls, regs, kpts, strides):
            B, _, H, W = c.shape

            # Classification Scores -> Apply Sigmoid
            # Shape: (B, H * W, n_classes)
            scores = c.permute(0, 2, 3, 1).reshape(B, H * W, -1).sigmoid()
            
            anchors = make_anchors(H, W, c.device, c.dtype).unsqueeze(0)            
            # Decode Bounding Boxes
            # Distribution Focal Loss (DFL) decoding: (B, 4 * reg_max, H, W)
            reg_max = r.shape[1] // 4
            r_dist = r.view(B, 4, reg_max, H * W).permute(
                0, 3, 1, 2
            )  # (B, H*W, 4, reg_max)
            r_dist = F.softmax(r_dist, dim=-1)
            proj = torch.arange(reg_max, device=r.device, dtype=r.dtype)
            dist = (r_dist * proj).sum(
                dim=-1
            )  # (B, H*W, 4) -> [left, top, right, bottom]

            # Convert distances (l, t, r, b) from anchor center into (x1, y1, x2, y2) image coordinates
            x1y1 = (anchors - dist[..., :2]) * stride
            x2y2 = (anchors + dist[..., 2:]) * stride
            boxes = torch.cat([x1y1, x2y2], dim=-1)  # (B, H*W, 4)

            # Decode Keypoints
            # k shape: (B, num_kpts * 3, H, W) -> [dx, dy, vis]
            num_kpts = k.shape[1] // 3
            k_reshaped = k.permute(0, 2, 3, 1).reshape(B, H * W, num_kpts, 3)

            # Keypoint coordinates scaled to image pixels with grid offsets
            kpt_xy = (anchors.unsqueeze(2) + k_reshaped[..., :2] * 2.0) * stride
            kpt_vis = k_reshaped[..., 2:].sigmoid()  # Keypoint visibility score
            keypoints = torch.cat([kpt_xy, kpt_vis], dim=-1)  # (B, H*W, num_kpts, 3)

            decoded_boxes.append(boxes)
            decoded_scores.append(scores)
            decoded_keypoints.append(keypoints)

        # Concatenate across all feature map scales
        final_boxes = torch.cat(decoded_boxes, dim=1)  # (B, Total_Anchors, 4)
        final_scores = torch.cat(decoded_scores, dim=1)  # (B, Total_Anchors, n_classes)
        final_kpts = torch.cat(
            decoded_keypoints, dim=1
        )  # (B, Total_Anchors, num_kpts, 3)

        return {"boxes": final_boxes, "scores": final_scores, "keypoints": final_kpts}
