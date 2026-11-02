import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.box_util import bbox_iou


class BoxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def _ciou_loss(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        fg_mask: torch.Tensor,
        target_scores_sum: float,
    ) -> torch.Tensor:
        """
        Computes normalized CIoU regression loss.

        Parameters
        ----------
        pred_boxes : torch.Tensor
            Predicted boxes. Shape: (B, num_pred_boxes, 4)
        target_boxes : torch.Tensor
            Target boxes. Shape: (B, num_pred_boxes, 4)
        fg_mask : torch.Tensor
            Boolean mask indicating foreground anchors. Shape: (B, num_pred_boxes)
        target_scores_sum : float
            Sum of target scores for normalization.
        """
        if not fg_mask.any():
            return pred_boxes.sum() * 0.0

        pred_boxes_fg = pred_boxes[fg_mask]
        target_boxes_fg = target_boxes[fg_mask]

        ious = bbox_iou(pred_boxes_fg, target_boxes_fg, c_iou=True)
        loss_iou = 1.0 - ious
        return loss_iou.sum() / target_scores_sum

    def _dfl_loss(
        self,
        pred_dist: torch.Tensor,
        target_ltrb: torch.Tensor,
        fg_mask: torch.Tensor,
        target_scores_sum: float,
    ) -> torch.Tensor:
        """
        Computes normalized Distribution Focal Loss (DFL).

        Parameters
        ----------
        pred_dist : torch.Tensor
            Predicted box distribution logits. Shape: (B, num_pred_boxes, 4 * reg_max)
        target_ltrb : torch.Tensor
            Target distance to the 4 edges (left, top, right, bottom) from anchor points.
            Shape: (B, num_pred_boxes, 4)
        fg_mask : torch.Tensor
            Boolean mask indicating foreground anchors. Shape: (B, num_pred_boxes)
        target_scores_sum : float
            Sum of target scores for normalization.
        """
        if not fg_mask.any():
            return pred_dist.sum() * 0.0

        B, N, _ = pred_dist.shape
        reg_max = pred_dist.shape[-1] // 4

        # Reshape to separate the 4 box coordinates from the reg_max bins
        pred_dist = pred_dist.view(B, N, 4, reg_max)

        # Filter for foreground anchors only
        pred_dist_fg = pred_dist[fg_mask]  # Shape: (num_fg, 4, reg_max)
        target_ltrb_fg = target_ltrb[fg_mask]  # Shape: (num_fg, 4)

        # Clamp targets to valid bin boundaries
        target_ltrb_fg = target_ltrb_fg.clamp(0, reg_max - 1.01)

        # Find lower and upper integer bin indices for linear interpolation weights
        lt = target_ltrb_fg.long()  # Floor index
        rt = lt + 1  # Ceil index
        weight_l = rt.float() - target_ltrb_fg
        weight_r = target_ltrb_fg - lt.float()

        # Compute cross-entropy loss over the distribution bins for both bounds
        loss_l = F.cross_entropy(
            pred_dist_fg.reshape(-1, reg_max), lt.reshape(-1), reduction="none"
        ).view(-1, 4)

        loss_r = F.cross_entropy(
            pred_dist_fg.reshape(-1, reg_max), rt.reshape(-1), reduction="none"
        ).view(-1, 4)

        # Combine losses using linear weights and average across the 4 edges
        dfl = (loss_l * weight_l + loss_r * weight_r).mean(dim=-1)

        return dfl.sum() / target_scores_sum

    def forward(
        self,
        pred_boxes: torch.Tensor,
        pred_dist: torch.Tensor,
        target_boxes: torch.Tensor,
        target_ltrb: torch.Tensor,
        fg_mask: torch.Tensor,
        target_scores_sum: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes total box loss (CIoU + DFL), scaled by respective gains.
        """
        ciou_loss = self._ciou_loss(pred_boxes, target_boxes, fg_mask, target_scores_sum)
        dfl_loss = self._dfl_loss(pred_dist, target_ltrb, fg_mask, target_scores_sum)

        return ciou_loss, dfl_loss
