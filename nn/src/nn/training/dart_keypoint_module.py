import math

import pytorch_lightning as pl
import torch

from nn.data.datamodule import Batch
from nn.model.yolo import DecodedPredictions, TrainingPredictions, YoloV10
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
    ):
        super().__init__()

        self.model = model
        self.model_config = model_config
        self.optimizer_config = optimizer_config or OptimizerConfig()
        self.loss = loss

    def forward(self, images: torch.Tensor) -> TrainingPredictions | DecodedPredictions:
        return self.model(images)

    def common_step(self, batch: Batch, batch_idx, stage: str):
        outputs: TrainingPredictions = self(batch.images)

        one2one, one2many = outputs.one2one, outputs.one2many

        anchors = torch.cat(self.model.anchors(batch.images.shape[2:], batch.images.device), dim=0)  # pyright: ignore[reportArgumentType]
        strides = torch.cat(self.model.strides(batch.images.shape[2:], batch.images.device), dim=0)  # pyright: ignore[reportArgumentType]

        loss_o2m, num_assignment_o2m = self.loss(batch, one2many, anchors, strides, one2one=False)
        loss_o2o, num_assignment_o2o = self.loss(batch, one2one, anchors, strides, one2one=True)

        loss = loss_o2m["loss"] + loss_o2o["loss"]

        for key, value in loss_o2m.items():
            self.log(
                f"loss/{stage}_o2m_{key}",
                value,
                on_step=True,
                on_epoch=True,
            )
        for key, value in loss_o2o.items():
            self.log(
                f"loss/{stage}_o2o_{key}",
                value,
                on_step=True,
                on_epoch=True,
            )

        self.log(
            f"assignment/{stage}_o2m",
            num_assignment_o2m,
            on_step=False,
            on_epoch=True,
        )

        self.log(
            f"assignment/{stage}_o2o",
            num_assignment_o2o,
            on_step=False,
            on_epoch=True,
        )

        return loss

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
