import math

import pytorch_lightning as pl
import torch

from nn.data.augmentation import TrainAugmentation
from nn.data.config import AugmentConfig
from nn.data.datamodule import Batch
from nn.metrics.map import MeanAveragePrecision
from nn.model.decode import flatten_head_outputs
from nn.model.yolo import YoloOutput, YoloV10
from nn.training.config import (
    ModelConfig,
    OptimizerConfig,
)


class DartKeypointModule(pl.LightningModule):
    def __init__(
        self,
        model: YoloV10,
        loss: torch.nn.Module,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig | None = None,
        augment_config: AugmentConfig | None = None,
    ):
        super().__init__()

        self.model = model
        self.model_config = model_config
        self.optimizer_config = optimizer_config or OptimizerConfig()
        self.loss = loss
        self.augment = (
            TrainAugmentation(augment_config)
            if augment_config is not None and augment_config.enabled
            else None
        )
        self.epoch_average_precision = {
            "val": MeanAveragePrecision(),
            "test": MeanAveragePrecision(),
        }

    def forward(self, images: torch.Tensor) -> YoloOutput:
        return self.model(images)

    def augment_batch(self, batch: Batch) -> Batch:
        """
        Apply the training augmentations on the device of the batch.

        Parameters
        ----------
        batch : Batch
            The batch as it was transferred to the training device.

        Returns
        -------
        Batch
            The augmented batch, or the unchanged batch when augmentation is disabled.
        """
        if self.augment is None:
            return batch

        images, gt_boxes, gt_labels, gt_keypoints = self.augment(
            batch.images, batch.gt_boxes, batch.gt_labels, batch.gt_keypoints
        )

        return batch._replace(
            images=images, gt_boxes=gt_boxes, gt_labels=gt_labels, gt_keypoints=gt_keypoints
        )

    def common_step(self, batch: Batch, batch_idx, stage: str):
        if stage == "train":
            batch = self.augment_batch(batch)

        outputs = self(batch.images)

        anchors = torch.cat(self.model.anchors(batch.images.shape[2:], batch.images.device), dim=0)  # pyright: ignore[reportArgumentType]
        strides = torch.cat(self.model.strides(batch.images.shape[2:], batch.images.device), dim=0)  # pyright: ignore[reportArgumentType]

        one2one = flatten_head_outputs(*outputs.one2one)
        loss_o2o, num_assignment_o2o = self.loss(batch, one2one, anchors, strides, one2one=True)
        loss = loss_o2o["loss"]

        for key, value in loss_o2o.items():
            self.log(f"loss/{stage}_o2o_{key}", value, on_step=True, on_epoch=True)
        self.log(f"assignment/{stage}_o2o", num_assignment_o2o, on_step=False, on_epoch=True)

        if outputs.one2many is not None:
            one2many = flatten_head_outputs(*outputs.one2many)
            loss_o2m, num_assignment_o2m = self.loss(
                batch, one2many, anchors, strides, one2one=False
            )
            loss = loss + loss_o2m["loss"]

            for key, value in loss_o2m.items():
                self.log(f"loss/{stage}_o2m_{key}", value, on_step=True, on_epoch=True)
            self.log(f"assignment/{stage}_o2m", num_assignment_o2m, on_step=False, on_epoch=True)

        if stage in self.epoch_average_precision and outputs.predictions is not None:
            self.epoch_average_precision[stage].update(
                outputs.predictions.boxes,
                outputs.predictions.scores,
                batch.gt_boxes,
                batch.gt_labels,
            )

        return loss

    def _log_average_precision(self, stage: str) -> None:
        metric = self.epoch_average_precision[stage]
        for name, value in metric.compute().items():
            self.log(f"metrics/{stage}_{name}", value, prog_bar=name == "mAP50")
        metric.reset()

    def on_validation_epoch_end(self) -> None:
        self._log_average_precision("val")

    def on_test_epoch_end(self) -> None:
        self._log_average_precision("test")

    def training_step(self, batch: Batch, batch_idx):
        return self.common_step(batch, batch_idx, stage="train")

    def validation_step(self, batch: Batch, batch_idx):
        return self.common_step(batch, batch_idx, stage="val")

    def test_step(self, batch: Batch, batch_idx):
        return self.common_step(batch, batch_idx, stage="test")

    def configure_optimizers(self):
        config = self.optimizer_config
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        if config.scheduler != "onecycle":
            return optimizer

        total_steps = self.trainer.estimated_stepping_batches
        if not math.isfinite(total_steps):
            return optimizer

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.lr,
            total_steps=max(1, int(total_steps)),
            pct_start=config.warmup_ratio,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
