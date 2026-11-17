from typing import NamedTuple

import torch

from nn.model.decode import DecodedPredictions, decode_raw_predictions, make_anchors
from nn.model.parts import YoloParts


class YoloOutput(NamedTuple):
    """
    Output of :class:`YoloV10`.

    Attributes
    ----------
    one2one : tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
        Raw per-scale head predictions from the one-to-one head, consumed by the loss
        after flattening with ``flatten_head_outputs``.
    one2many : tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]] | None
        Raw per-scale head predictions from the auxiliary one-to-many head, ``None`` when
        the model was built without it.
    predictions : DecodedPredictions | None
        Decoded predictions of the one-to-one head in image pixels. Only computed in eval
        mode, where they are needed for metrics; ``None`` while training.
    """

    one2one: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
    one2many: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]] | None
    predictions: DecodedPredictions | None


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

    def forward(self, x) -> YoloOutput:
        assert isinstance(x, torch.Tensor), (
            f"Expected input to be a torch.Tensor, but got {type(x)}"
        )

        backbone_feature_maps = self.backbone(x)

        neck_features = self.neck(backbone_feature_maps)

        one2one = self.head([feature.detach() for feature in neck_features])
        one2many = None if self.aux_head is None else self.aux_head(neck_features)

        return YoloOutput(
            one2one=one2one,
            one2many=one2many,
            predictions=None
            if self.training
            else decode_raw_predictions(*one2one, self.backbone.get_strides()),
        )

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
        Returns the strides for each scale, flattened to one entry per anchor.

        Parameters
        ----------
        img_size : tuple[int, int]
            The size of the input image (height, width).
        device : torch.device
            The device on which the strides should be created.

        Returns
        -------
        list[torch.Tensor]
            A list containing the strides for each scale of shape (h * w, 2).
        """

        return [
            torch.full(
                (img_size[0] // stride * img_size[1] // stride, 2),
                stride,
                device=device,
                dtype=torch.float32,
            )
            for stride in self.backbone.get_strides()
        ]
