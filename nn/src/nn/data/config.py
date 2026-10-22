from dataclasses import dataclass


@dataclass(frozen=True)
class DataModuleConfig:
    """
    Configuration for the datamodule
    """

    batch_size: int = 64
    num_workers: int = 8
    prefetch_factor: int = 1
    pin_memory: bool = False
