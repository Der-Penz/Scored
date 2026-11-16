from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class Transform(ABC):
    @abstractmethod
    def __call__(
        self,
        X: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_keypoints: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def train_only(self) -> bool:
        pass


class Resize(Transform):
    """
    Resize the image to a fixed pixel size and scale the targets accordingly.

    Parameters
    ----------
    img_size : tuple[int, int]
        Target ``(height, width)`` of the resized image.
    """

    def __init__(self, img_size: tuple[int, int]):
        self.img_size = img_size

    def train_only(self) -> bool:
        return False

    def __call__(
        self,
        X: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_keypoints: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_h, target_w = self.img_size
        _, h, w = X.shape

        if (h, w) == (target_h, target_w):
            return X, gt_boxes, gt_labels, gt_keypoints

        X = F.interpolate(
            X.unsqueeze(0), size=(target_h, target_w), mode="bilinear", align_corners=False
        ).squeeze(0)

        # both targets live in pixels, so they only need per-axis scaling
        scale = torch.tensor([target_w / w, target_h / h], dtype=gt_boxes.dtype)
        gt_boxes = gt_boxes * torch.cat([scale, scale])
        gt_keypoints = gt_keypoints * torch.tensor(
            [scale[0], scale[1], 1.0], dtype=gt_keypoints.dtype
        )

        return X, gt_boxes, gt_labels, gt_keypoints
