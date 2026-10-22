from dataclasses import dataclass

import pytorch_lightning as pl
from lightning_fabric.plugins.precision.precision import _PRECISION_INPUT


@dataclass(frozen=True)
class ModelConfig:
    n_classes: int
    num_kpts: int
    reg_max: int = 16
    include_aux_head: bool = True


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 0.01
    scheduler: str = "onecycle"  # "onecycle" or "none"
    warmup_ratio: float = 0.1


@dataclass(frozen=True)
class AssignmentConfig:
    alpha: float = 0.5
    beta: float = 6.0
    o2m_topk: int = 10


@dataclass(frozen=True)
class TrainerConfig:
    max_epochs: int = 100
    accelerator: str = "auto"
    devices: str | int = "auto"
    precision: _PRECISION_INPUT = "bf16-mixed"
    log_every_n_steps: int = 10
    gradient_clip_val: float | None = 10.0
    num_sanity_val_steps: int = 0
    default_root_dir: str | None = None
    fast_dev_run: bool = False

    def build(self) -> pl.Trainer:
        return pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            devices=self.devices,
            precision=self.precision,
            log_every_n_steps=self.log_every_n_steps,
            gradient_clip_val=self.gradient_clip_val,
            num_sanity_val_steps=self.num_sanity_val_steps,
            default_root_dir=self.default_root_dir,
            fast_dev_run=self.fast_dev_run,
        )
