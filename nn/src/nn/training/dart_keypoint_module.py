import math

import pytorch_lightning as pl
import torch

from nn.data.datamodule import Batch
from nn.loss.assigner import assign_box
from nn.model.yolo import DecodedPredictions, TrainingPredictions, YoloV10
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

    def forward(self, images: torch.Tensor) -> TrainingPredictions | DecodedPredictions:
        return self.model(images)

    def compute_loss(
        self,
        batch: Batch,
        head_outputs: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]],
        one2one: bool,
    ) -> tuple[torch.Tensor, float]:
        cls_preds, reg_preds, kpt_preds = head_outputs
        decoded_preds = self.model.decode_raw_predictions(
            cls_preds, reg_preds, kpt_preds, self.model.backbone.get_strides()
        )

        anchors = torch.cat(self.model.anchors(batch.images.shape[2:], batch.images.device), dim=0)  # pyright: ignore[reportArgumentType]
        assignment = assign_box(
            batch.gt_boxes,
            batch.gt_labels,
            pred_boxes=decoded_preds["boxes"],
            pred_scores=decoded_preds["scores"],
            anchors=anchors,
            alpha=self.assignment_config.alpha,
            beta=self.assignment_config.beta,
            topk=1 if one2one else self.assignment_config.o2m_topk,
        )

        return self.loss(
            decoded_preds, batch.gt_boxes, batch.gt_labels, batch.gt_keypoints, assignment
        ), assignment[3].sum().item()

    def common_step(self, batch: Batch, batch_idx, stage: str):
        outputs: TrainingPredictions = self(batch.images)

        one2one, one2many = outputs.one2one, outputs.one2many

        loss_o2m, assignment_o2m = self.compute_loss(batch, one2many, one2one=False)
        loss_o2o, assignment_o2o = self.compute_loss(batch, one2one, one2one=True)

        loss = loss_o2m + loss_o2o

        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            f"{stage}_loss_o2m",
            loss_o2m,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            f"{stage}_loss_o2o",
            loss_o2o,
            on_step=True,
            on_epoch=True,
        )

        self.log(
            f"{stage}_assignment_o2m",
            assignment_o2m,
            on_step=False,
            on_epoch=True,
        )

        self.log(
            f"{stage}_assignment_o2o",
            assignment_o2o,
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
