import torch
from nn.box_util import bbox_iou


def assign_box(
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    anchors: torch.Tensor,
    alpha: float,
    beta: float,
    topk: int,
):
    r"""
    Assign predicted boxes to ground truth boxes based on a combination
    of classification score and IoU.
    The matching score is calculated as:
    :math:`m = s * p^alpha * IoU^beta`.

    Parameters
    ----------
    gt_boxes : torch.Tensor
        Ground truth boxes for the current batch. Shape: (N, max_num_boxes, 4)
    gt_labels : torch.Tensor
        Ground truth labels for the current batch. Shape: (N, max_num_boxes)
    pred_boxes : torch.Tensor
        Concatenated predicted boxes across all scales for the current batch.
        Shape: (N, num_pred_boxes, 4)
    pred_scores : torch.Tensor
        Concated predicted scores across all scales for the current batch.
        Shape: (N, num_pred_boxes, num_classes)
    anchors : torch.Tensor
        Anchor boxes across all scales in the same order as pred_<> in p1 coordinates.
        Shape: (N, num_pred_boxes, 2)
    alpha : float
        Weight exponent for the classification score $p$
    beta : float
        Weight exponent for the IoU score
    topk : int
        Number of top-k predictions to consider for assignment

    Returns
    -------
    target_labels : torch.Tensor
        Target labels for each predicted box. Shape: (N, num_pred_boxes)
    target_boxes : torch.Tensor
        Target boxes for each predicted box. Shape: (N, num_pred_boxes, 4)
    target_scores : torch.Tensor
        Target scores for each predicted box. Shape: (N, num_pred_boxes, num_classes)
    fg_mask : torch.Tensor
        Boolean mask indicating foreground anchors. Shape: (N, num_pred_boxes)
    """
    gt_mask = (gt_labels >= 0).float().unsqueeze(-1)  # (B, M, 1)

    # compute s to get boolean mask for valid predictions per anchor
    pred_cls_scores = compute_pred_scores(pred_scores, gt_labels)  # (B, M, N)
    ious = bbox_iou(gt_boxes, pred_boxes)  # (B, M, N)
    is_in_gts = compute_anchor_containment_mask(gt_boxes, anchors)  # (B, M, N)

    scores = is_in_gts.float() * (pred_cls_scores**alpha) * (ious**beta) * gt_mask  # (B, M, N)

    # Top-k selection per ground truth box
    _, topk_indices = torch.topk(scores, topk, dim=-1)  # (B, M, topk)
    topk_mask = torch.zeros_like(scores, dtype=torch.bool)
    topk_mask.scatter_(-1, topk_indices, True)

    mask_pos = topk_mask & (scores > 0)  # (B, M, N)

    # Resolve conflicts where one anchor is assigned to multiple GT boxes by keeping
    # the one with the highest score
    overlaps = mask_pos.sum(dim=1)  # (B, N)
    if (overlaps > 1).any():
        _, max_gt_idx = scores.max(dim=1)  # (B, N)
        mask_pos = torch.zeros_like(scores, dtype=torch.bool)
        mask_pos.scatter_(1, max_gt_idx.unsqueeze(1), True)
        mask_pos = mask_pos & (overlaps.unsqueeze(1) > 0)

    # Determine foreground anchors
    fg_mask = mask_pos.any(dim=1)  # (B, N)

    # Extract target GT per anchor
    target_gt_idx = mask_pos.long().argmax(dim=1)  # (B, N)
    batch_idx = torch.arange(gt_boxes.shape[0], device=gt_boxes.device).unsqueeze(1)

    assigned_gt_labels = gt_labels[batch_idx, target_gt_idx]  # (B, N)
    target_boxes = gt_boxes[batch_idx, target_gt_idx]  # (B, N, 4)

    # Zero-out targets for background predictions
    target_labels = torch.where(fg_mask, assigned_gt_labels, -1)
    target_boxes = target_boxes * fg_mask.unsqueeze(-1)

    # Construct soft classification target normalized by max score
    max_align_metric = scores.max(dim=-1, keepdim=True)[0].clamp(min=1e-6)  # (B, M, 1)
    norm_align_metric = ((scores / max_align_metric) * mask_pos.float()).sum(dim=1)  # (B, N)

    target_scores = torch.zeros_like(pred_scores)  # (B, N, num_classes)
    if fg_mask.any():
        target_scores.scatter_(
            -1, target_labels.clamp(min=0).unsqueeze(-1), norm_align_metric.unsqueeze(-1)
        )
    target_scores = target_scores * fg_mask.unsqueeze(-1)

    return target_labels, target_boxes, target_scores, fg_mask


def compute_pred_scores(pred_scores: torch.Tensor, gt_labels: torch.Tensor) -> torch.Tensor:
    """
    Retrieve the prediction class score matrix based on the gt

    Parameters
    ----------
    pred_scores : torch.Tensor
        Predicted class probabilities/logits per anchor. Shape: (B, N, C)
    gt_labels : torch.Tensor
        Ground truth labels for the current batch. Shape: (B, M)

    Returns
    -------
    torch.Tensor
        A tensor of shape (B, M, N) containing the predicted scores for each GT label
        across all anchors.
    """

    B, M = gt_labels.shape
    N = pred_scores.shape[1]

    # Convert negative labels to 0 temporarily for indexing
    valid_gt_labels = torch.where(
        gt_labels >= 0, gt_labels, torch.tensor(0, device=gt_labels.device)
    )

    # Expand shapes so every GT label aligns with every predicted anchor
    gt_labels_expanded = (
        valid_gt_labels.unsqueeze(-1).unsqueeze(-1).expand(B, M, N, 1)
    )  # shape (B, M, N, 1)

    # Expand pred_scores to match the shape of gt_labels_expanded
    pred_scores_expanded = pred_scores.unsqueeze(1).expand(B, M, N, -1)  # shape (B, M, N, C)

    # Gather the score for each GT label across the class dimension
    matched_scores = torch.gather(pred_scores_expanded, dim=3, index=gt_labels_expanded).squeeze(
        -1
    )  # shape (B, M, N)

    # Apply gt_mask so padded GTs (-1) output score 0.0
    gt_mask = (gt_labels >= 0).float().unsqueeze(-1)
    return matched_scores * gt_mask  # shape (B, M, N)


def compute_anchor_containment_mask(gt_boxes, anchors):
    """
    gt_boxes : torch.Tensor
            Ground truth boxes for the current batch. Shape: (B, max_num_boxes (M), 4)
    anchors : torch.Tensor
            Anchor boxes across all scales in the same order as pred_<> in p1 coordinates.
            Shape: (num_pred_boxes (N), 2)

    Returns:
        is_in_gts: Boolean Tensor of shape (B, M, N) indicating if anchor n is inside GT m.
    """
    anchor_x, anchor_y = anchors[None, None, :, 0], anchors[None, None, :, 1]
    x1, y1, x2, y2 = gt_boxes[..., 0:1], gt_boxes[..., 1:2], gt_boxes[..., 2:3], gt_boxes[..., 3:4]

    is_in_gts = (anchor_x > x1) & (anchor_x < x2) & (anchor_y > y1) & (anchor_y < y2)
    return is_in_gts  # shape (B, M, N)
