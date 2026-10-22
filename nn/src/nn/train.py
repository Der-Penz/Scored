import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import tyro

from nn.data.datamodule import DartKeyPointDataModule, DataModuleConfig
from nn.data.dataset import DatasetConfig
from nn.loss.config import LossConfig
from nn.loss.loss import YoloV10Loss
from nn.model.impl.simple import MobileYoloHead, MobileYoloNeck, TimmYoloBackbone
from nn.model.parts import YoloParts
from nn.model.yolo import YoloV10
from nn.training.config import AssignmentConfig, ModelConfig, OptimizerConfig, TrainerConfig
from nn.training.dart_keypoint_module import DartKeypointModule


def load_dataset_config_from_json(dataset_config_path: Path) -> DatasetConfig:
    with open(dataset_config_path) as f:
        data = json.load(f)
    return DatasetConfig(**data)


JsonDatasetConfig = Annotated[
    DatasetConfig,
    tyro.conf.arg(
        constructor=load_dataset_config_from_json,
        help="Path to the dataset configuration JSON file",
    ),
]


@dataclass
class TrainCLIArgs:
    dataset: JsonDatasetConfig

    model: ModelConfig
    optimizer: OptimizerConfig = OptimizerConfig()
    assignment: AssignmentConfig = AssignmentConfig()
    trainer: TrainerConfig = TrainerConfig()
    datamodule: DataModuleConfig = DataModuleConfig()
    loss: LossConfig = LossConfig()

    backbone_model_name: str = "mobilenetv4_conv_small.e1200_r224_in1k"
    pretrained: bool = True


def main() -> None:
    args = tyro.cli(TrainCLIArgs)

    transforms = []
    datamodule = DartKeyPointDataModule(args.datamodule, args.dataset, transforms)

    parts = YoloParts(
        backbone=lambda: TimmYoloBackbone(args.backbone_model_name, pretrained=args.pretrained),
        neck=lambda channels: MobileYoloNeck(channels),
        aux_head=lambda n_classes, num_kpts, reg_max, channels: MobileYoloHead(
            n_classes, num_kpts, reg_max, channels
        ),
        head=lambda n_classes, num_kpts, reg_max, channels: MobileYoloHead(
            n_classes, num_kpts, reg_max, channels
        ),
    )
    model = YoloV10(
        n_classes=args.model.n_classes,
        num_kpts=args.model.num_kpts,
        reg_max=args.model.reg_max,
        parts=parts,
        include_aux_head=args.model.include_aux_head,
    )
    loss = YoloV10Loss(args.loss, args.model)

    module = DartKeypointModule(
        model=model,
        loss=loss,
        model_config=args.model,
        optimizer_config=args.optimizer,
    )

    trainer = args.trainer.build()
    trainer.fit(module, datamodule=datamodule)

    trainer.test(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
