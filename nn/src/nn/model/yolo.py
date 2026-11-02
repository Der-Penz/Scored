from typing import NamedTuple

import torch
from nn.model.decode import DecodedPredictions, decode_raw_predictions, make_anchors
from nn.model.parts import YoloParts


class TrainingPredictions(NamedTuple):
    one2one: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
    one2many: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]


class YoloV10(torch.nn.Module):
    def __init__(
        self,
        n_classes: int,
        num_kpts: int,
        reg_max: int,
        parts: YoloParts,
        include_aux_head: bool = True,
    ):
        super().__init__()
        (
            self.backbone,
            self.neck,
            self.head,
            self.aux_head,
        ) = parts.build_parts(n_classes, num_kpts, reg_max, include_aux_head=include_aux_head)

    def forward(self, x) -> TrainingPredictions | DecodedPredictions:
        assert isinstance(x, torch.Tensor), (
            f"Expected input to be a torch.Tensor, but got {type(x)}"
        )

        backbone_feature_maps = self.backbone(x)

        neck_features = self.neck(backbone_feature_maps)

        if self.aux_head is None:  # Inference-only model
            assert self.training is False, (
                "Model should be in eval mode during inference when no aux_head is provided."
            )
            cls, reg, kpts = self.head(neck_features)
            return decode_raw_predictions(cls, reg, kpts, self.backbone.get_strides())

        one2one = self.head([feature.detach() for feature in neck_features])
        one2many = self.aux_head(neck_features)

        return TrainingPredictions(one2one, one2many)

    def num_boxes_per_scale(self, img_size: tuple[int, int]) -> tuple[int, ...]:
        """
        Returns the number of boxes per scale.

        Parameters
        ----------
        img_size : tuple[int, int]
            The size of the input image (height, width).
        Returns
        -------
        tuple[int, ...]
            A tuple containing the number of boxes per scale.
        """
        strides = self.backbone.get_strides()

        return tuple(img_size[0] // s * img_size[1] // s for s in strides)

    def anchors(self, img_size: tuple[int, int], device: torch.device) -> list[torch.Tensor]:
        """
        Returns the anchor boxes for each scale.

        Parameters
        ----------
        img_size : tuple[int, int]
            The size of the input image (height, width).
        device : torch.device
            The device on which the anchor boxes should be created.

        Returns
        -------
        tuple[torch.Tensor, ...]
            A tuple containing the anchor boxes for each scale.
        """
        return [
            make_anchors(
                img_size[0] // stride,
                img_size[1] // stride,
                device=device,
                dtype=torch.float32,
            )
            for stride in self.backbone.get_strides()
        ]

    def strides(self, img_size: tuple[int, int], device: torch.device) -> list[torch.Tensor]:
        """
        Returns the strides for each scale.

        Returns
        -------
        list[torch.Tensor]
            A list containing the strides for each scale.
        """

        return [
            torch.full(
                (img_size[0] // stride, img_size[1] // stride, 2),
                stride,
                device=device,
                dtype=torch.float32,
            )
            for stride in self.backbone.get_strides()
        ]
