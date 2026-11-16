from dataclasses import dataclass


@dataclass(frozen=True)
class DataModuleConfig:
    """
    Configuration for the datamodule.

    Parameters
    ----------
    batch_size : int
        Number of images per batch.
    num_workers : int
        Number of dataloader workers.
    prefetch_factor : int
        Number of batches each worker prefetches.
    pin_memory : bool
        Whether the dataloader pins memory before transferring to the device.
    img_size : int | tuple[int, int]
        Input resolution the images and targets are resized to. A single integer yields a
        square input of that size, a ``(height, width)`` tuple a rectangular one.
    """

    batch_size: int = 64
    num_workers: int = 8
    prefetch_factor: int = 1
    pin_memory: bool = False
    img_size: int | tuple[int, int] = 640

    @property
    def image_size(self) -> tuple[int, int]:
        """
        The input resolution as ``(height, width)``.

        Returns
        -------
        tuple[int, int]
            The height and width every image is resized to.
        """
        height, width = (
            (self.img_size, self.img_size) if isinstance(self.img_size, int) else self.img_size
        )
        if height <= 0 or width <= 0:
            raise ValueError(f"img_size must be positive, got {self.img_size}")

        return height, width
