from collections.abc import Sequence

import torch
from torchvision.ops import box_iou, nms


class MeanAveragePrecision:
    """
    Box mean average precision in the style of COCO.

    Every detection is matched greedily to the ground truth boxes of its class. Precision and
    recall are evaluated at 10 IoU thresholds from 0.5 to 0.95 and integrated with the 101
    point interpolation. Predictions are first filtered by confidence and reduced by
    class-wise non-maximum suppression, so the metric scores the detections an inference
    pipeline would produce.

    The metric accumulates over calls to :meth:`update` and has to be :meth:`reset` after
    every epoch.

    Parameters
    ----------
    iou_thresholds : Sequence[float] | None, optional
        IoU thresholds a detection has to reach to count as a true positive. Defaults to
        ``0.5, 0.55, ..., 0.95``.
    conf_threshold : float, optional
        Minimum class score a detection needs to enter the metric.
    nms_iou : float, optional
        IoU threshold of the class-wise non-maximum suppression.
    max_detections : int, optional
        Maximum number of detections kept per image and class after suppression.
    """

    def __init__(
        self,
        iou_thresholds: Sequence[float] | None = None,
        conf_threshold: float = 0.001,
        nms_iou: float = 0.7,
        max_detections: int = 300,
    ):
        self.iou_thresholds = (
            torch.linspace(0.5, 0.95, 10)
            if iou_thresholds is None
            else torch.tensor(iou_thresholds, dtype=torch.float32)
        )
        self.conf_threshold = conf_threshold
        self.nms_iou = nms_iou
        self.max_detections = max_detections
        self.reset()

    def reset(self) -> None:
        """Drop all accumulated detections and ground truth boxes."""
        self._scores: dict[int, list[torch.Tensor]] = {}
        self._true_positives: dict[int, list[torch.Tensor]] = {}
        self._num_gt: dict[int, int] = {}

    @torch.no_grad()
    def update(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> None:
        """
        Add one batch of predictions and ground truth boxes.

        Parameters
        ----------
        boxes : torch.Tensor
            Predicted boxes in pixels in (x1, y1, x2, y2) format, shape (B, N, 4).
        scores : torch.Tensor
            Class probabilities of the predicted boxes, shape (B, N, num_classes).
        gt_boxes : torch.Tensor
            Ground truth boxes in pixels, shape (B, M, 4). Padding boxes are ignored.
        gt_labels : torch.Tensor
            Class labels of the ground truth boxes, shape (B, M), ``-1`` marks padding.
        """
        boxes = boxes.detach().cpu()
        scores = scores.detach().cpu()
        gt_boxes = gt_boxes.detach().cpu()
        gt_labels = gt_labels.detach().cpu()

        for image in range(boxes.shape[0]):
            valid = gt_labels[image] >= 0
            image_gt = gt_boxes[image][valid]
            labels = gt_labels[image][valid]
            for label in labels.tolist():
                self._num_gt[label] = self._num_gt.get(label, 0) + 1

            for label in range(scores.shape[-1]):
                det_boxes, det_scores = self._detect(boxes[image], scores[image, :, label])
                self._scores.setdefault(label, []).append(det_scores)
                self._true_positives.setdefault(label, []).append(
                    self._match(det_boxes, image_gt[labels == label])
                )

    def compute(self) -> dict[str, float]:
        """
        Compute the mean average precision over everything added since the last reset.

        Returns
        -------
        dict[str, float]
            ``mAP50`` at IoU 0.5 and ``mAP50-95`` averaged over all IoU thresholds, both
            averaged over the classes that have ground truth boxes. Both are ``0.0`` when no
            ground truth or no detections were accumulated.
        """
        average_precisions: list[torch.Tensor] = []

        for label, num_gt in self._num_gt.items():
            empty = torch.zeros(self.iou_thresholds.numel())
            if label not in self._scores:
                average_precisions.append(empty)
                continue

            scores = torch.cat(self._scores[label])
            if scores.numel() == 0:
                average_precisions.append(empty)
                continue

            true_positives = torch.cat(self._true_positives[label])
            order = scores.argsort(descending=True)
            average_precisions.append(self._average_precision(true_positives[order], num_gt))

        if not average_precisions:
            return {"mAP50": 0.0, "mAP50-95": 0.0}

        average_precisions = torch.stack(average_precisions)
        return {
            "mAP50": average_precisions[:, 0].mean().item(),
            "mAP50-95": average_precisions.mean().item(),
        }

    def _detect(
        self, boxes: torch.Tensor, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Filter the predictions of one image and class by confidence and run NMS.

        Parameters
        ----------
        boxes : torch.Tensor
            Predicted boxes of one class, shape (N, 4).
        scores : torch.Tensor
            Predicted scores of one class, shape (N,).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The kept boxes and their scores, highest score first.
        """
        keep = scores > self.conf_threshold
        boxes, scores = boxes[keep], scores[keep]
        if boxes.shape[0] == 0:
            return boxes, scores

        keep = nms(boxes, scores, self.nms_iou)[: self.max_detections]
        return boxes[keep], scores[keep]

    def _match(self, det_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        """
        Match the detections of one image and class to its ground truth boxes.

        Parameters
        ----------
        det_boxes : torch.Tensor
            Detections sorted by descending score, shape (N, 4).
        gt_boxes : torch.Tensor
            Ground truth boxes of the class, shape (M, 4).

        Returns
        -------
        torch.Tensor
            True positive flags, shape (N, num_thresholds).
        """
        matches = torch.zeros((det_boxes.shape[0], self.iou_thresholds.numel()), dtype=torch.bool)
        if det_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
            return matches

        ious = box_iou(det_boxes, gt_boxes)  # (N, M)

        for index, threshold in enumerate(self.iou_thresholds):
            pairs = torch.nonzero(ious >= threshold)  # (K, 2), ordered by detection
            if pairs.shape[0] == 0:
                continue
            pairs = pairs[ious[pairs[:, 0], pairs[:, 1]].argsort(descending=True)]
            # every ground truth box and every detection may only be matched once
            pairs = pairs[_first_occurrence(pairs[:, 1], gt_boxes.shape[0])]
            pairs = pairs[_first_occurrence(pairs[:, 0], det_boxes.shape[0])]
            matches[pairs[:, 0], index] = True

        return matches

    def _average_precision(self, true_positives: torch.Tensor, num_gt: int) -> torch.Tensor:
        """
        Integrate the precision/recall curve of every IoU threshold with 101 points.

        Parameters
        ----------
        true_positives : torch.Tensor
            True positive flags sorted by descending score, shape (N, num_thresholds).
        num_gt : int
            Number of ground truth boxes of the class.

        Returns
        -------
        torch.Tensor
            Average precision per IoU threshold, shape (num_thresholds,).
        """
        true_positives = true_positives.float()
        tp_cumulative = true_positives.cumsum(dim=0)
        fp_cumulative = (1.0 - true_positives).cumsum(dim=0)
        recall = tp_cumulative / max(num_gt, 1)
        precision = tp_cumulative / (tp_cumulative + fp_cumulative).clamp(min=1.0)

        points = torch.linspace(0.0, 1.0, 101)
        average_precision = torch.zeros(true_positives.shape[1])

        for index in range(true_positives.shape[1]):
            # precision envelope of the curve, including the fixed end points
            envelope = torch.cat([torch.ones(1), precision[:, index], torch.zeros(1)])
            envelope = envelope.flip(0).cummax(dim=0).values.flip(0)
            recall_curve = torch.cat([torch.zeros(1), recall[:, index], torch.ones(1)])

            interpolated = torch.searchsorted(recall_curve, points)
            interpolated = interpolated.clamp(max=envelope.numel() - 1)
            average_precision[index] = envelope[interpolated].mean()

        return average_precision


def _first_occurrence(ids: torch.Tensor, num_ids: int) -> torch.Tensor:
    """
    Return the indices of the first occurrence of every id.

    Parameters
    ----------
    ids : torch.Tensor
        Ids of the box pairs, ordered from the highest to the lowest IoU.
    num_ids : int
        Number of possible ids.

    Returns
    -------
    torch.Tensor
        Sorted indices of the first occurrence of each id, ids that never occur are dropped.
    """
    sentinel = ids.shape[0]
    first = torch.full((num_ids,), sentinel, dtype=torch.long)
    first.scatter_(0, ids.flip(0), torch.arange(sentinel).flip(0))
    return first[first < sentinel].sort().values
