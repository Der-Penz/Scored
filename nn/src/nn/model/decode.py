from typing import NamedTuple

import torch

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


def flatten_head_outputs(
    cls: list[torch.Tensor],
    regs: list[torch.Tensor],
    kpts: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flattens the per-scale head feature maps into anchor-major tensors.

    The scales are concatenated in the same order as :func:`decode_raw_predictions`, so
    anchor ``n`` of the returned tensors corresponds to ``anchors[n]`` built from the same
    scales.

    Parameters
    ----------
    cls : list[torch.Tensor]
        Classification feature maps per scale. Shape: (B, n_classes, H_i, W_i)
    regs : list[torch.Tensor]
        Box regression feature maps per scale. Shape: (B, 4 * reg_max, H_i, W_i)
    kpts : list[torch.Tensor]
        Keypoint feature maps per scale. Shape: (B, num_kpts * 3, H_i, W_i)

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Flattened ``cls``, ``regs`` and ``kpts`` tensors of shape (B, Total_Anchors, C).
    """

    def flatten(maps: list[torch.Tensor]) -> torch.Tensor:
        batch = maps[0].shape[0]
        return torch.cat(
            [m.permute(0, 2, 3, 1).reshape(batch, -1, m.shape[1]) for m in maps],
            dim=1,
        )

    return flatten(cls), flatten(regs), flatten(kpts)


def decode_flat_predictions(
    cls: torch.Tensor,
    regs: torch.Tensor,
    kpts: torch.Tensor,
    anchors: torch.Tensor,
    strides: torch.Tensor | None = None,
    sigmoid_visibility: bool = True,
) -> DecodedPredictions:
    """
    Decodes flattened head predictions into bounding boxes, class scores, and keypoints.

    Parameters
    ----------
    cls : torch.Tensor
        Flattened classification logits. Shape: (B, Total_Anchors, n_classes)
    regs : torch.Tensor
        Flattened box regression logits. Shape: (B, Total_Anchors, 4 * reg_max)
    kpts : torch.Tensor
        Flattened keypoint predictions. Shape: (B, Total_Anchors, num_kpts * 3)
    anchors : torch.Tensor
        Anchor points matching each prediction. Shape: (Total_Anchors, 2)
    strides : torch.Tensor | None, optional
        Per-anchor strides of shape (Total_Anchors, 2). When ``None`` the predictions are
        returned in anchor/grid space, which is what the losses work in.
    sigmoid_visibility : bool, optional
        Whether to squash keypoint visibility through a sigmoid. Leave enabled for
        inference (visibility as probabilities), disable when the output feeds a
        logits-based loss such as ``BCEWithLogitsLoss``.

    Returns
    -------
    DecodedPredictions
        Decoded predictions containing:
        - "boxes": (B, Total_Anchors, 4) in (x1, y1, x2, y2) space
        - "scores": (B, Total_Anchors, n_classes) class probabilities
        - "keypoints": (B, Total_Anchors, num_kpts, 3) in (x, y, visibility) space
    """
    B, N, _ = cls.shape
    anchors = anchors.to(device=cls.device, dtype=cls.dtype)

    scores = cls.sigmoid()

    # Distribution Focal Loss (DFL) decoding: (B, N, 4 * reg_max)
    reg_max = regs.shape[-1] // 4
    reg_dist = regs.view(B, N, 4, reg_max).softmax(dim=-1)
    proj = torch.arange(reg_max, device=regs.device, dtype=regs.dtype)
    dist = (reg_dist * proj).sum(dim=-1)  # (B, N, 4) -> [left, top, right, bottom]

    # Convert distances (l, t, r, b) from the anchor center into (x1, y1, x2, y2)
    boxes = torch.cat(
        [anchors.unsqueeze(0) - dist[..., :2], anchors.unsqueeze(0) + dist[..., 2:]], dim=-1
    )

    # kpts shape: (B, N, num_kpts * 3) -> [dx, dy, vis]
    num_kpts = kpts.shape[-1] // 3
    kpt_reshaped = kpts.view(B, N, num_kpts, 3)
    kpt_xy = anchors.unsqueeze(0).unsqueeze(2) + kpt_reshaped[..., :2] * 2.0
    kpt_vis = kpt_reshaped[..., 2:]
    if sigmoid_visibility:
        kpt_vis = kpt_vis.sigmoid()

    if strides is not None:
        strides = strides.to(device=cls.device, dtype=cls.dtype)
        boxes = boxes * torch.cat([strides, strides], dim=-1).unsqueeze(0)
        kpt_xy = kpt_xy * strides.unsqueeze(0).unsqueeze(2)

    keypoints = torch.cat([kpt_xy, kpt_vis], dim=-1)  # (B, N, num_kpts, 3)

    return DecodedPredictions(boxes=boxes, scores=scores, keypoints=keypoints)


def decode_raw_predictions(
    cls: list[torch.Tensor],
    regs: list[torch.Tensor],
    kpts: list[torch.Tensor],
    strides: tuple[int, ...],
    sigmoid_visibility: bool = True,
) -> DecodedPredictions:
    """
    Decodes raw multi-scale head predictions into bounding boxes, class scores, and keypoints.

    Parameters
    ----------
    cls : list[torch.Tensor]
        Classification feature maps per scale. Shape: (B, n_classes, H_i, W_i)
    regs : list[torch.Tensor]
        Box regression feature maps per scale. Shape: (B, 4 * reg_max, H_i, W_i)
    kpts : list[torch.Tensor]
        Keypoint feature maps per scale. Shape: (B, num_kpts * 3, H_i, W_i)
    strides : tuple[int, ...]
        Strides corresponding to each feature map scale.
    sigmoid_visibility : bool, optional
        Whether to squash keypoint visibility through a sigmoid. Leave enabled for
        inference (visibility as probabilities), disable when the output feeds a
        logits-based loss such as ``BCEWithLogitsLoss``.

    Returns
    -------
    DecodedPredictions
        Decoded predictions, see :func:`decode_flat_predictions`.
    """
    anchors = torch.cat(
        [make_anchors(c.shape[2], c.shape[3], c.device, c.dtype) for c in cls], dim=0
    )
    stride_tensor = torch.cat(
        [
            torch.full((c.shape[2] * c.shape[3], 2), s, device=c.device, dtype=c.dtype)
            for c, s in zip(cls, strides, strict=True)
        ],
        dim=0,
    )

    return decode_flat_predictions(
        *flatten_head_outputs(cls, regs, kpts),
        anchors=anchors,
        strides=stride_tensor,
        sigmoid_visibility=sigmoid_visibility,
    )
