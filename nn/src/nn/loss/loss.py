import torch.nn as nn

from nn.loss.config import LossConfig
from nn.training.config import ModelConfig


class YoloV10Loss(nn.Module):
    def __init__(
        self,
        loss_config: LossConfig,
        model_config: ModelConfig,
    ):
        super().__init__()
        self.loss_config = loss_config
        self.model_config = model_config
