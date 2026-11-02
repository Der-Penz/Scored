from dataclasses import dataclass


@dataclass(frozen=True)
class WeightsConfig:
    box: float = 7.5
    kpt_pos: float = 12.0
    kpt_vis: float = 1.0
    dfl: float = 1.5
    cls: float = 0.5


@dataclass(frozen=True)
class AssignmentConfig:
    alpha: float = 0.5
    beta: float = 6.0
    o2m_topk: int = 10


@dataclass(frozen=True)
class LossConfig:
    weights: WeightsConfig = WeightsConfig()
    assignment: AssignmentConfig = AssignmentConfig()
    sigmas: list[float] = [0.025, 0.04]
