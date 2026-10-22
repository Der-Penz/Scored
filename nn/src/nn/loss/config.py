from dataclasses import dataclass


@dataclass(frozen=True)
class WeightsConfig:
    box: float = 1.0
    keypoint: float = 1.0


@dataclass(frozen=True)
class LossConfig:
    weights: WeightsConfig = WeightsConfig()
