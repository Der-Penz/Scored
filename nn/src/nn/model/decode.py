from typing import NamedTuple

import torch
import torch.nn.functional as F
from nn.box_util import XYXYBox


class DecodedPredictions(NamedTuple):
    boxes: XYXYBox  # (B, Total_Anchors, 4)
    scores: torch.Tensor  # (B, Total_Anchors, n_classes)
    keypoints: torch.Tensor  # (B, Total_Anchors, num_kpts, 3)


def make_anchors(
    h: int, w: int, device: str | torch.device, dtype: torch.dtype, offset: float = 0.5
) -> torch.Tensor:
    """Generate anchor points for a grid of size (h, w).

    Parameters
    ----------
    h : int
        Height of the grid.
    w : int
        Width of the grid.
    device : str | torch.device
        Device to place the anchor points on.
    dtype : torch.dtype
        Data type of the anchor points.
    offset : float, optional
        Offset for the anchor points, by default 0.5.

    Returns
    -------
    torch.Tensor
        A tensor of shape (h * w, 2) containing the anchor points in the shape [H, W, 2] where the
        last dimension represents (x, y) coordinates.
    """
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype) + offset,
        torch.arange(w, device=device, dtype=dtype) + offset,
        indexing="ij",
    )
    anchors = torch.stack((grid_x, grid_y), dim=-1)
    return anchors.view(-1, 2)


def decode_raw_predictions(
    cls: list[torch.Tensor],
    regs: list[torch.Tensor],
    kpts: list[torch.Tensor],
    strides: tuple[int, ...],
) -> DecodedPredictions:
    """
    Decodes raw multi-scale head predictions into bounding boxes, class scores, and keypoints.

    Parameters
    ----------
    cls : list[torch.Tensor]
        Classification feature maps per scale. Shape: (B, n_classes, H_i, W_i)
    regs : list[torch.Tensor]
        Box regression feature maps per scale.
        Shape: (B, 4, H_i, W_i) or (B, 4 * reg_max, H_i, W_i)
    kpts : list[torch.Tensor]
        Keypoint feature maps per scale. Shape: (B, num_kpts * 3, H_i, W_i)
    strides : tuple[int, ...]
        Strides corresponding to each feature map scale.

    Returns
    -------
    DecodedPredictions
        Dictionary containing:
        - "boxes": (B, Total_Anchors, 4) in (x1, y1, x2, y2) pixel space
        - "scores": (B, Total_Anchors, n_classes) class probabilities
        - "keypoints": (B, Total_Anchors, num_kpts, 3) in (x, y, visibility) pixel space
    """

    decoded_boxes = []
    decoded_scores = []
    decoded_keypoints = []

    for c, r, k, stride in zip(cls, regs, kpts, strides, strict=False):
        B, _, H, W = c.shape

        # Classification Scores -> Apply Sigmoid
        # Shape: (B, H * W, n_classes)
        scores = c.permute(0, 2, 3, 1).reshape(B, H * W, -1).sigmoid()

        anchors = make_anchors(H, W, c.device, c.dtype).unsqueeze(0)
        # Decode Bounding Boxes
        # Distribution Focal Loss (DFL) decoding: (B, 4 * reg_max, H, W)
        reg_max = r.shape[1] // 4
        r_dist = r.view(B, 4, reg_max, H * W).permute(0, 3, 1, 2)  # (B, H*W, 4, reg_max)
        r_dist = F.softmax(r_dist, dim=-1)
        proj = torch.arange(reg_max, device=r.device, dtype=r.dtype)
        dist = (r_dist * proj).sum(dim=-1)  # (B, H*W, 4) -> [left, top, right, bottom]

        # Convert distances (l, t, r, b) from anchor center into (x1, y1, x2, y2)
        # image coordinates
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
    final_kpts = torch.cat(decoded_keypoints, dim=1)  # (B, Total_Anchors, num_kpts, 3)

    return DecodedPredictions(boxes=final_boxes, scores=final_scores, keypoints=final_kpts)
