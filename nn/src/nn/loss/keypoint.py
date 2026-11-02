import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointLoss(nn.Module):
    def __init__(self, sigmas: list[float]):
        super().__init__()

        self.register_buffer("sigmas", torch.tensor(sigmas))

    def forward(
        self,
        pred_kpts: torch.Tensor,  # (B, N, K, 3)
        target_kpts: torch.Tensor,  # (B, N, K, 3)
        target_boxes: torch.Tensor,  # (B, N, 4)
        fg_mask: torch.Tensor,  # (B, N)
        target_scores_sum: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not fg_mask.any():
            return pred_kpts.sum() * 0.0, pred_kpts.sum() * 0.0

        # Filter foreground instances
        pk = pred_kpts[fg_mask]  # (num_fg, K, 2 or 3)
        tk = target_kpts[fg_mask]  # (num_fg, K, 3)
        tb = target_boxes[fg_mask]  # (num_fg, 4)

        pred_xy = pk[..., :2]
        gt_xy = tk[..., :2]
        vis = tk[
            ..., 2
        ]  # (num_fg, K) 0 for unlabeled, 1 for labeled or 1 for occluded and 2 for visible
        vis_mask = vis > 0  # Only loss on labeled keypoints

        # Object scale s = sqrt(w * h)
        box_w = (tb[..., 2] - tb[..., 0]).clamp(min=1e-6)
        box_h = (tb[..., 3] - tb[..., 1]).clamp(min=1e-6)
        scale = torch.sqrt(box_w * box_h).unsqueeze(-1)  # (num_fg, 1)

        # Squared Euclidean distance
        d2 = torch.sum((pred_xy - gt_xy) ** 2, dim=-1)  # (num_fg, K)

        # OKS calculation
        sigmas_sq = (2 * self.sigmas**2).unsqueeze(0)  # type: ignore # (1, K)
        oks = torch.exp(-d2 / (2 * (scale**2) * sigmas_sq + 1e-9))

        # OKS Loss: 1 - OKS
        kpt_loss = (1.0 - oks) * vis_mask
        loss_kpt = kpt_loss.sum() / (vis_mask.sum() + 1e-9)

        # Visibility score loss (BCE)
        pred_vis_logits = pk[..., 2]
        target_vis_targets = (vis > 0).float()
        loss_vis = (
            F.binary_cross_entropy_with_logits(pred_vis_logits, target_vis_targets, reduction="sum")
            / target_scores_sum
        )

        return loss_kpt, loss_vis
