import math

import pytorch_lightning as pl
import torch

from nn.model.yolo import YoloV10
from nn.training.config import (
    AssignmentConfig,
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
        assignment_config: AssignmentConfig | None = None,
    ):
        super().__init__()

        self.model = model
        self.model_config = model_config
        self.optimizer_config = optimizer_config or OptimizerConfig()
        self.loss = loss
        self.assignment_config = assignment_config or AssignmentConfig()

    def forward(self, images: torch.Tensor):
        return self.model(images)

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
