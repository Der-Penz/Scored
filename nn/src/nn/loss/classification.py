import torch


class ClassificationLoss(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred_scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        """
        Computes the normalized BCE classification loss for YOLO.

        Parameters
        ----------
        pred_scores : torch.Tensor
            Predicted classification scores (logits). Shape: (B, num_pred_boxes, num_classes)
        target_scores : torch.Tensor
            Soft target scores from assign_box. Shape: (B, num_pred_boxes, num_classes)

        Returns
        -------
        loss : torch.Tensor
            Scalar classification loss.
        """
        target_scores_dtype = target_scores.to(dtype=pred_scores.dtype)

        # Compute total weight/sum of target scores for normalization across the batch
        target_scores_sum = max(target_scores.sum().item(), 1.0)

        loss = self.bce(pred_scores, target_scores_dtype)

        return loss.sum() / target_scores_sum
