from abc import ABC, abstractmethod

import torch


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
