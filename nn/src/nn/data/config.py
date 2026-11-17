from dataclasses import dataclass


@dataclass(frozen=True)
class AugmentConfig:
    """
    Configuration for the training-time augmentations.

    All augmentations run on the device of the batch, so they execute on the GPU during
    training. Every entry with a probability of ``0`` is disabled.

    Parameters
    ----------
    enabled : bool
        Whether to augment the training batches at all.
    mosaic_p : float
        Probability of building a 2x2 mosaic from the batch for a sample. The anchor image
        always occupies the top-left tile, the other three tiles are sampled at random from
        the same batch.
    mixup_p : float
        Probability of blending a sample with another sample of the batch. The blend ratio is
        drawn from a Beta distribution, the targets of both samples are kept.
    mixup_alpha : float
        Both parameters of the Beta distribution the blend ratio is drawn from. Larger values
        concentrate the ratio around 0.5.
    hflip_p : float
        Probability of a horizontal flip.
    vflip_p : float
        Probability of a vertical flip. Keep at ``0`` for a camera that is always mounted
        in the same orientation.
    affine_p : float
        Probability of a random rotation, translation, scale and shear.
    degrees : float
        Maximum absolute rotation in degrees.
    translate : float
        Maximum absolute translation as a fraction of the image size.
    scale : tuple[float, float]
        Lower and upper bound of the random scaling factor.
    shear : float
        Maximum absolute shear in degrees.
    color_jitter_p : float
        Probability of a random brightness, contrast, saturation and hue shift.
    brightness : float
        Maximum absolute brightness shift.
    contrast : float
        Maximum absolute contrast shift.
    saturation : float
        Maximum absolute saturation shift.
    hue : float
        Maximum absolute hue shift.
    grayscale_p : float
        Probability of converting an image to grayscale.
    blur_p : float
        Probability of blurring an image.
    blur_kernel_size : int
        Odd kernel size of the Gaussian blur.
    blur_sigma : tuple[float, float]
        Lower and upper bound of the Gaussian blur sigma.
    """

    enabled: bool = True
    mosaic_p: float = 0.5
    mixup_p: float = 0.1
    mixup_alpha: float = 32.0
    hflip_p: float = 0.5
    vflip_p: float = 0.0
    affine_p: float = 0.5
    degrees: float = 10.0
    translate: float = 0.1
    scale: tuple[float, float] = (0.8, 1.2)
    shear: float = 0.0
    color_jitter_p: float = 0.5
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.3
    hue: float = 0.05
    grayscale_p: float = 0.2
    blur_p: float = 0.1
    blur_kernel_size: int = 5
    blur_sigma: tuple[float, float] = (0.1, 1.5)


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
