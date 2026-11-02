from typing import TypedDict

import torch

from nn.loss.assigner import assign_box
from nn.loss.box import BoxLoss
from nn.loss.classification import ClassificationLoss
from nn.loss.config import LossConfig
from nn.loss.keypoint import KeypointLoss
from nn.model.decode import decode_raw_predictions
from nn.training.dart_keypoint_module import Batch


class LossParts(TypedDict):
    loss: torch.Tensor
    cls_loss: torch.Tensor
    box_loss: torch.Tensor
    dfl_loss: torch.Tensor
    kpt_pos_loss: torch.Tensor
    kpt_vis_loss: torch.Tensor


class YoloV10Loss(torch.nn.Module):
    def __init__(
        self,
        loss_config: LossConfig,
    ):
        super().__init__()
        self.loss_config = loss_config

        self.classification_loss = ClassificationLoss()
        self.box_loss = BoxLoss()
        self.keypoint_loss = KeypointLoss(loss_config.sigmas)

    def forward(
        self,
        batch: Batch,
        head_outputs: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]],
        anchors: torch.Tensor,
        strides: torch.Tensor,
        one2one: bool,
    ) -> tuple[LossParts, int]:
        """
        Main forward pass for YOLOv10 loss calculation.

        Parameters
        ----------
        batch : Batch
            A batch of data containing ground truth boxes, labels, and other relevant information.
        head_outputs : tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
            Tuple containing lists of predicted classification scores, regression outputs,
            and keypoint predictions.
        anchors : torch.Tensor
            The anchor boxes for each box in the shape of (num_pred_boxes, 4).
        strides : torch.Tensor
            The strides value for each box in the shape of (num_pred_boxes, 2).
        one2one : bool
            Whether to use one-to-one assignment or one-to-many assignment.
        """
        cls_preds, reg_preds, kpt_preds = head_outputs

        gt_boxes_stride_space = YoloV10Loss.scale_gt_to_stride_space(batch.gt_boxes, strides)

        stride_space_preds = decode_raw_predictions(
            cls_preds, reg_preds, kpt_preds, (1,) * len(cls_preds), sigmoid_visibility=False
        )  # we pass stride 1 to keep
        # boxes in stride space for assignment and loss calculation. Visibility stays as
        # logits because the keypoint loss consumes it with BCEWithLogitsLoss.

        # Assignment is a target-generation step: it must not backprop into the heads.
        with torch.no_grad():
            target_labels, target_boxes, target_scores, target_gt_idx, fg_mask = assign_box(
                gt_boxes_stride_space,
                batch.gt_labels,
                pred_boxes=stride_space_preds.boxes,
                pred_scores=stride_space_preds.scores,
                anchors=anchors,
                alpha=self.loss_config.assignment.alpha,
                beta=self.loss_config.assignment.beta,
                topk=1 if one2one else self.loss_config.assignment.o2m_topk,
            )

        target_scores_sum = max(target_scores.sum().item(), 1.0)

        cls_loss = self.classification_loss(cls_preds, target_scores)

        target_ltrb_stride_space = YoloV10Loss.lrtb_from_boxes(target_boxes, anchors)

        ciou_loss, dfl_loss = self.box_loss(
            pred_boxes=stride_space_preds.boxes,
            pred_dist=reg_preds,
            target_boxes=target_boxes,
            target_ltrb=target_ltrb_stride_space,
            fg_mask=fg_mask,
            target_scores_sum=target_scores_sum,
        )

        target_kpts_stride_space = YoloV10Loss.prepare_target_keypoints_from_gt(
            batch.gt_keypoints, strides, target_gt_idx
        )

        kpt_pos_loss, kpt_vis_loss = self.keypoint_loss(
            stride_space_preds.keypoints,
            target_kpts=target_kpts_stride_space,
            target_boxes=target_boxes,
            fg_mask=fg_mask,
            target_scores_sum=target_scores_sum,
        )

        total_loss = (
            cls_loss * self.loss_config.weights.cls
            + ciou_loss * self.loss_config.weights.box
            + dfl_loss * self.loss_config.weights.dfl
            + kpt_pos_loss * self.loss_config.weights.kpt_pos
            + kpt_vis_loss * self.loss_config.weights.kpt_vis
        )

        loss_items: LossParts = {
            "loss": total_loss,
            "cls_loss": cls_loss,
            "box_loss": ciou_loss,
            "kpt_pos_loss": kpt_pos_loss,
            "kpt_vis_loss": kpt_vis_loss,
            "dfl_loss": dfl_loss,
        }

        return loss_items, fg_mask.sum().item()  # type: ignore

    @staticmethod
    def scale_gt_to_stride_space(target_boxes: torch.Tensor, strides: torch.Tensor) -> torch.Tensor:
        """
        Scales assigned ground truth target boxes from absolute image space
        to per-anchor stride space using 2-value (X, Y) strides.

        Parameters
        ----------
        target_boxes : torch.Tensor
            Assigned target boxes in absolute image pixel coordinates.
            Shape: (B, num_pred_boxes, 4) [x1, y1, x2, y2]
        strides : torch.Tensor
            The pre-concatenated strides tensor matching each predicted box,
            containing [stride_x, stride_y] for each anchor.
            Shape: (num_pred_boxes, 2) or broadcastable.

        Returns
        -------
        torch.Tensor
            Scaled target boxes. Shape: (B, num_pred_boxes, 4)
        """
        if strides.shape[-1] == 2:
            stride_x = strides[..., 0:1]
            stride_y = strides[..., 1:2]
            strides_4d = torch.cat([stride_x, stride_y, stride_x, stride_y], dim=-1)
        else:
            strides_4d = strides.repeat(1, 4) if strides.ndim > 1 else strides.repeat(4, 1).T

        strides_expanded = strides_4d.unsqueeze(0)  # Shape: (1, N, 4)

        # Scale down coordinates: [x1/sx, y1/sy, x2/sx, y2/sy]
        scaled_boxes = target_boxes / strides_expanded

        return scaled_boxes

    @staticmethod
    def lrtb_from_boxes(target_boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """
        Converts target boxes from (x1, y1, x2, y2) format to (left, top, right, bottom)
        distances from anchor points.

        Parameters
        ----------
        target_boxes : torch.Tensor
            Target boxes in (x1, y1, x2, y2) format in stride space. Shape: (B, num_pred_boxes, 4)
        anchors : torch.Tensor
            Anchor points corresponding to each predicted box. Shape: (num_pred_boxes, 2)

        Returns
        -------
        torch.Tensor
            Distances to the left, top, right, and bottom edges of the target boxes
            from the anchor points. Shape: (B, num_pred_boxes, 4)
        """
        anchor_centers = anchors.unsqueeze(0)  # Shape: (1, N, 2)

        left = anchor_centers[..., 0] - target_boxes[..., 0]  # x1
        top = anchor_centers[..., 1] - target_boxes[..., 1]  # y1
        right = target_boxes[..., 2] - anchor_centers[..., 0]  # x2
        bottom = target_boxes[..., 3] - anchor_centers[..., 1]  # y2

        ltrb = torch.stack([left, top, right, bottom], dim=-1)  # Shape: (B, N, 4)

        return ltrb.clamp(min=0.0)  # ensure positive distances

    @staticmethod
    def prepare_target_keypoints_from_gt(
        gt_keypoints: torch.Tensor, strides: torch.Tensor, target_gt_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Prepares the gt keypoints for loss calculation by scaling them to stride space and
        aligning them with the predicted anchors.

        Parameters
        ----------
        gt_keypoints : torch.Tensor
            Ground truth keypoints for the current batch. Shape: (B, M, K, 3)
        strides : torch.Tensor
            The pre-concatenated strides tensor matching each predicted box,
            containing [stride_x, stride_y] for each anchor.
            Shape: (num_pred_boxes, 2) or broadcastable.
        target_gt_idx : torch.Tensor
            Index of the assigned ground truth box (0 to M-1) for each anchor.
            Shape: (B, N)

        Returns
        -------
        torch.Tensor
            Scaled target keypoints. Shape: (B, num_pred_boxes, K, 3)
        """
        # Scale the target keypoints to the stride space
        if strides.shape[-1] == 2:
            stride_x = strides[..., 0:1]
            stride_y = strides[..., 1:2]
            strides_3d = torch.cat(
                [stride_x, stride_y, torch.ones_like(stride_x)], dim=-1
            )  # (N, 3)
        else:
            strides_3d = strides.repeat(1, 3) if strides.ndim > 1 else strides.repeat(3, 1).T

        strides_expanded = strides_3d.unsqueeze(0).unsqueeze(2)  # (1, N, 1, 3)

        target_keypoints = gt_keypoints.gather(
            dim=1,
            index=target_gt_idx.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, gt_keypoints.shape[2], 3),
        )  # (B, N, K, 3)

        target_scaled_keypoints = target_keypoints / strides_expanded

        return target_scaled_keypoints
