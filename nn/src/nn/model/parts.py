from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn


class YoloBackbone(nn.Module, ABC):
    """
    A yolo styled Backbone for all passes scales
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_backbone_channels(self) -> tuple[int, ...]:
        """
        Returns the number of channels in each feature map produced by the backbone.

        Returns
        -------
        tuple[int, ...]
            A tuple containing the number of channels in each feature map produced by the backbone.
        """

    @abstractmethod
    def get_strides(self) -> tuple[int, ...]:
        """
        Returns the strides of the feature maps produced by the backbone.

        Returns
        -------
        tuple[int, ...]
            A tuple containing the strides of the feature maps produced by the backbone.
        """

    def __call__(self, x: torch.Tensor) -> list[torch.Tensor]:
        return super().__call__(x)


class YoloNeck(nn.Module, ABC):
    """
    A yolo styled Neck for all passes scales

    Parameters
    ----------
    channels : tuple[int, ...]
        The number of channels in each feature map coming from the backbone
    """

    def __init__(self, channels: tuple[int, ...]):
        super().__init__()
        self.channels = channels

    @abstractmethod
    def get_neck_channels(self) -> tuple[int, ...]:
        """
        Returns the number of channels in each feature map produced by the neck.

        Returns
        -------
        tuple[int, ...]
            A tuple containing the number of channels in each feature map produced by the neck.
        """

    def __call__(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        return super().__call__(x)


class YoloHead(nn.Module, ABC):
    """
    A yolo styled Head for all passes scales.

    Parameters
    ----------
    n_classes : int
        The number of classes to predict
    num_kpts : int
        The number of keypoints to predict
    channels : tuple[int, ...]
        The number of channels in each feature map

    Notes
    -----
    This is a base class for all yolo styled heads.
    """

    def __init__(self, n_classes: int, num_kpts: int, channels: tuple[int, ...]):
        super().__init__()

        self.n_classes = n_classes
        self.num_kpts = num_kpts
        self.channels = channels

    @abstractmethod
    def forward(
        self, x: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Forward pass of the head.

        Parameters
        ----------
        x : list[torch.Tensor]
            A list of feature maps from the neck. Each feature map should be a tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
            A tuple containing three lists: cls, reg, kpts predictions. Each list contains the predictions for each scale starting from the lowest stride to the highest stride. Each prediction is a tensor of shape (batch_size, [attributes], height, width).
        """

    def __call__(
        self, x: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        return super().__call__(x)


type BackboneFactory = Callable[[], YoloBackbone]
type NeckFactory = Callable[[tuple[int, ...]], YoloNeck]
type HeadFactory = Callable[[int, int, tuple[int, ...]], YoloHead]


@dataclass(frozen=True)
class YoloParts:
    backbone: BackboneFactory
    neck: NeckFactory
    head: HeadFactory  # Main one-to-one head
    aux_head: HeadFactory | None = (
        None  # Auxiliary one-to-many head only used for training
    )

    def build_parts(
        self,
        n_classes: int,
        num_kpts: int,
        include_aux_head: bool,
    ) -> tuple[YoloBackbone, YoloNeck, YoloHead, YoloHead | None]:
        backbone = self.backbone()

        backbone_channels = backbone.get_backbone_channels()
        neck = self.neck(backbone_channels)

        neck_channels = neck.get_neck_channels()

        head = self.head(n_classes, num_kpts, neck_channels)
        aux_head = None
        if include_aux_head and self.aux_head is not None:
            aux_head = self.aux_head(n_classes, num_kpts, neck_channels)
        return backbone, neck, head, aux_head
