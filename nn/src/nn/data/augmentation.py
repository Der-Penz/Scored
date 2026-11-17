import torch
import torch.nn.functional as F
from kornia import augmentation as K
from kornia.geometry.keypoints import Keypoints
from nn.data.config import AugmentConfig


class TrainAugmentation(torch.nn.Module):
    """
    GPU-side augmentation of training batches.

    Images, boxes and keypoints are already tensors in pixel space when they arrive here,
    so everything runs on the device of the batch and stays on the GPU. Boxes and keypoints
    are transformed with the same random parameters as the image through kornia's
    ``AugmentationSequential``.

    The batch layout follows :class:`nn.data.datamodule.Batch`: padded boxes, labels with
    ``-1`` for padding and keypoints of shape (B, M, num_keypoints, 3) with the last channel
    holding ``(x, y, visible)``.

    Parameters
    ----------
    config : AugmentConfig
        Which augmentations to apply and how strong.
    """

    def __init__(self, config: AugmentConfig):
        super().__init__()
        self.config = config

        if config.blur_p > 0 and config.blur_kernel_size % 2 == 0:
            raise ValueError(
                f"blur_kernel_size must be odd for a centered Gaussian blur, "
                f"got {config.blur_kernel_size}"
            )

        geometric: list[torch.nn.Module] = []
        if config.hflip_p > 0:
            geometric.append(K.RandomHorizontalFlip(p=config.hflip_p))
        if config.vflip_p > 0:
            geometric.append(K.RandomVerticalFlip(p=config.vflip_p))
        if config.affine_p > 0:
            geometric.append(
                K.RandomAffine(
                    degrees=config.degrees,
                    translate=(config.translate, config.translate),
                    scale=config.scale,
                    shear=config.shear,
                    p=config.affine_p,
                )
            )

        self.geometric = (
            K.AugmentationSequential(
                *geometric, data_keys=["input", "keypoints"], same_on_batch=False
            )
            if geometric
            else None
        )

        photometric: list[torch.nn.Module] = []
        if config.color_jitter_p > 0:
            photometric.append(
                K.ColorJiggle(
                    brightness=config.brightness,
                    contrast=config.contrast,
                    saturation=config.saturation,
                    hue=config.hue,
                    p=config.color_jitter_p,
                )
            )
        if config.grayscale_p > 0:
            photometric.append(K.RandomGrayscale(p=config.grayscale_p))
        if config.blur_p > 0:
            photometric.append(
                K.RandomGaussianBlur(
                    (config.blur_kernel_size, config.blur_kernel_size),
                    config.blur_sigma,
                    p=config.blur_p,
                )
            )

        self.photometric = (
            K.AugmentationSequential(*photometric, data_keys=["input"], same_on_batch=False)
            if photometric
            else None
        )

    def forward(
        self,
        images: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_keypoints: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Augment a batch of images together with its boxes and keypoints.

        Parameters
        ----------
        images : torch.Tensor
            Batch of images of shape (B, 3, H, W).
        gt_boxes : torch.Tensor
            Boxes in pixels in (x1, y1, x2, y2) format of shape (B, M, 4).
        gt_labels : torch.Tensor
            Class labels of shape (B, M), ``-1`` marks padding.
        gt_keypoints : torch.Tensor
            Keypoints in pixels of shape (B, M, num_keypoints, 3), the last channel is
            ``(x, y, visible)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            The augmented images, boxes, labels and keypoints in the same layout as the
            inputs, except that the object axis grows to four times its size when a mosaic
            is built.
        """
        if not self.config.enabled:
            return images, gt_boxes, gt_labels, gt_keypoints

        if self.config.mosaic_p > 0:
            images, gt_boxes, gt_labels, gt_keypoints = self.mosaic(
                images, gt_boxes, gt_labels, gt_keypoints, self.config.mosaic_p
            )

        if self.geometric is not None and gt_boxes.shape[1] > 0:
            images, gt_boxes, gt_keypoints = self.apply_geometric(images, gt_boxes, gt_keypoints)

        if self.config.mixup_p > 0:
            images, gt_boxes, gt_labels, gt_keypoints = self.mixup(
                images,
                gt_boxes,
                gt_labels,
                gt_keypoints,
                self.config.mixup_p,
                self.config.mixup_alpha,
            )

        if self.photometric is not None:
            images = self.photometric(images)

        images = images.clamp(0.0, 1.0)

        return self.postprocess(images, gt_boxes, gt_labels, gt_keypoints)

    def apply_geometric(
        self,
        images: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_keypoints: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply the geometric transforms to the image, the boxes and the keypoints.

        The boxes do not go through kornia's box code path, whose pixel convention differs by
        one pixel from the keypoint one. Their four corners are transformed as additional
        keypoints instead, so boxes and keypoints always stay aligned, and the box is rebuilt
        as the tight bounding box of the transformed corners.

        Parameters
        ----------
        images : torch.Tensor
            Batch of images of shape (B, 3, H, W).
        gt_boxes : torch.Tensor
            Boxes in pixels in (x1, y1, x2, y2) format of shape (B, M, 4).
        gt_keypoints : torch.Tensor
            Keypoints in pixels of shape (B, M, num_keypoints, 3).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The transformed images, boxes and keypoints.
        """
        assert self.geometric is not None

        batch, num_objects = gt_boxes.shape[:2]
        num_keypoints = gt_keypoints.shape[2]

        corners = torch.stack(
            [
                gt_boxes[..., [0, 1]],
                gt_boxes[..., [2, 1]],
                gt_boxes[..., [2, 3]],
                gt_boxes[..., [0, 3]],
            ],
            dim=2,
        )
        points = torch.cat([gt_keypoints[..., :2], corners], dim=2).reshape(batch, -1, 2)

        images, points = self.geometric(images, Keypoints(points))
        points = points.to_tensor().reshape(batch, num_objects, num_keypoints + 4, 2)

        keypoints, corners = points[:, :, :num_keypoints], points[:, :, num_keypoints:]

        return (
            images,
            torch.cat([corners.min(dim=2).values, corners.max(dim=2).values], dim=-1),
            torch.cat([keypoints, gt_keypoints[..., 2:]], dim=-1),
        )

    @staticmethod
    def mosaic(
        images: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_keypoints: torch.Tensor,
        p: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build a 2x2 mosaic out of the images of the batch.

        Tile 0 always shows the anchor image itself, tiles 1 to 3 show random other images
        of the same batch. The object axis grows from M to 4 * M, with the targets of tile
        ``i`` placed in ``[i * M, (i + 1) * M)``. Boxes and keypoints are scaled and shifted
        into their tile, so they stay in pixel space of the composed image.

        Parameters
        ----------
        images : torch.Tensor
            Batch of images of shape (B, 3, H, W).
        gt_boxes : torch.Tensor
            Boxes in pixels in (x1, y1, x2, y2) format of shape (B, M, 4).
        gt_labels : torch.Tensor
            Class labels of shape (B, M), ``-1`` marks padding.
        gt_keypoints : torch.Tensor
            Keypoints in pixels of shape (B, M, num_keypoints, 3).
        p : float
            Probability that a sample is turned into a mosaic.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Images, boxes, labels and keypoints after the mosaic.
        """
        batch, _, height, width = images.shape
        num_objects = gt_boxes.shape[1]
        num_keypoints = gt_keypoints.shape[2]
        device = images.device

        if num_objects == 0:
            return images, gt_boxes, gt_labels, gt_keypoints

        mosaic_mask = torch.rand(batch, device=device) < p
        if not mosaic_mask.any():
            return images, gt_boxes, gt_labels, gt_keypoints

        canvas = images.clone()
        out_boxes = gt_boxes.new_zeros((batch, 4 * num_objects, 4))
        out_labels = gt_labels.new_full((batch, 4 * num_objects), -1)
        out_keypoints = gt_keypoints.new_zeros((batch, 4 * num_objects, num_keypoints, 3))

        half_width, half_height = width // 2, height // 2
        tiles = (
            (0, 0, half_width, half_height),
            (half_width, 0, width, half_height),
            (0, half_height, half_width, height),
            (half_width, half_height, width, height),
        )

        for tile, (x1, y1, x2, y2) in enumerate(tiles):
            tile_width, tile_height = x2 - x1, y2 - y1
            # the anchor image stays in the top left tile, the others are random partners
            indices = (
                torch.arange(batch, device=device)
                if tile == 0
                else torch.randint(0, batch, (batch,), device=device)
            )
            start, end = tile * num_objects, (tile + 1) * num_objects

            canvas[:, :, y1:y2, x1:x2] = F.interpolate(
                images[indices],
                size=(tile_height, tile_width),
                mode="bilinear",
                align_corners=False,
            )

            scale_x, scale_y = tile_width / width, tile_height / height
            out_boxes[:, start:end] = gt_boxes[indices] * gt_boxes.new_tensor(
                [scale_x, scale_y, scale_x, scale_y]
            ) + gt_boxes.new_tensor([x1, y1, x1, y1])
            out_labels[:, start:end] = gt_labels[indices]
            out_keypoints[:, start:end] = gt_keypoints[indices] * gt_keypoints.new_tensor(
                [scale_x, scale_y, 1.0]
            ) + gt_keypoints.new_tensor([x1, y1, 0.0])

        # samples that were not selected keep their original targets in the first tile
        keep_mask = mosaic_mask.view(batch, 1)
        out_labels[:, :num_objects] = torch.where(keep_mask, out_labels[:, :num_objects], gt_labels)
        out_boxes[:, :num_objects] = torch.where(
            keep_mask.unsqueeze(-1), out_boxes[:, :num_objects], gt_boxes
        )
        out_keypoints[:, :num_objects] = torch.where(
            keep_mask.view(batch, 1, 1, 1), out_keypoints[:, :num_objects], gt_keypoints
        )

        out_labels[~mosaic_mask, num_objects:] = -1
        out_boxes[~mosaic_mask, num_objects:] = 0
        out_keypoints[~mosaic_mask, num_objects:] = 0

        return (
            torch.where(mosaic_mask.view(batch, 1, 1, 1), canvas, images),
            out_boxes,
            out_labels,
            out_keypoints,
        )

    @staticmethod
    def mixup(
        images: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_keypoints: torch.Tensor,
        p: float,
        alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Blend a sample with another sample of the batch and keep the targets of both.

        The blend ratio is drawn per sample from ``Beta(alpha, alpha)`` and the partner is a
        different sample of the same batch. Since a blended image still shows the darts of
        both frames, the targets of both are kept, which grows the object axis from M to
        2 * M. Samples that are not blended get the partner targets as padding.

        Parameters
        ----------
        images : torch.Tensor
            Batch of images of shape (B, 3, H, W).
        gt_boxes : torch.Tensor
            Boxes in pixels in (x1, y1, x2, y2) format of shape (B, M, 4).
        gt_labels : torch.Tensor
            Class labels of shape (B, M), ``-1`` marks padding.
        gt_keypoints : torch.Tensor
            Keypoints in pixels of shape (B, M, num_keypoints, 3).
        p : float
            Probability that a sample is blended with another sample.
        alpha : float
            Both parameters of the Beta distribution the blend ratio is drawn from.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Images, boxes, labels and keypoints after the blend.
        """
        batch = images.shape[0]
        num_objects = gt_boxes.shape[1]

        if batch < 2 or num_objects == 0:
            return images, gt_boxes, gt_labels, gt_keypoints

        mixup_mask = torch.rand(batch, device=images.device) < p
        if not mixup_mask.any():
            return images, gt_boxes, gt_labels, gt_keypoints

        # offset by at least one so a sample never blends with itself
        offset = torch.randint(1, batch, (batch,), device=images.device)
        partner = (torch.arange(batch, device=images.device) + offset) % batch

        ratio = torch.distributions.Beta(alpha, alpha).sample((batch,)).to(images)
        ratio = torch.where(mixup_mask, ratio, torch.ones_like(ratio)).view(batch, 1, 1, 1)
        images = images * ratio + images[partner] * (1.0 - ratio)

        out_boxes = torch.cat([gt_boxes, gt_boxes[partner]], dim=1)
        out_labels = torch.cat([gt_labels, gt_labels[partner]], dim=1)
        out_keypoints = torch.cat([gt_keypoints, gt_keypoints[partner]], dim=1)

        # samples that were not blended must not gain the targets of their partner
        out_labels[~mixup_mask, num_objects:] = -1
        out_boxes[~mixup_mask, num_objects:] = 0
        out_keypoints[~mixup_mask, num_objects:] = 0

        return images, out_boxes, out_labels, out_keypoints

    @staticmethod
    def postprocess(
        images: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_keypoints: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Bring the augmented targets back into a valid batch layout.

        Boxes are clipped to the image, keypoints that were transformed out of the frame are
        clipped and marked invisible, and objects that fell off the image completely become
        padding again.

        Parameters
        ----------
        images : torch.Tensor
            The augmented images of shape (B, 3, H, W).
        gt_boxes : torch.Tensor
            Boxes in pixels in (x1, y1, x2, y2) format of shape (B, M, 4).
        gt_labels : torch.Tensor
            Class labels of shape (B, M), ``-1`` marks padding.
        gt_keypoints : torch.Tensor
            Keypoints in pixels of shape (B, M, num_keypoints, 3).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            The images together with clipped boxes, labels and keypoints.
        """
        height, width = images.shape[2], images.shape[3]

        boxes = torch.stack(
            [
                gt_boxes[..., 0].clamp(0.0, width),
                gt_boxes[..., 1].clamp(0.0, height),
                gt_boxes[..., 2].clamp(0.0, width),
                gt_boxes[..., 3].clamp(0.0, height),
            ],
            dim=-1,
        )

        keypoints = gt_keypoints.clone()
        x, y = keypoints[..., 0], keypoints[..., 1]
        inside = (x >= 0) & (x <= width) & (y >= 0) & (y <= height)
        keypoints[..., 0] = x.clamp(0.0, width)
        keypoints[..., 1] = y.clamp(0.0, height)
        keypoints[..., 2] = keypoints[..., 2] * inside

        # objects that are entirely outside the frame carry no supervision anymore
        keep = (gt_labels >= 0) & (boxes[..., 2] > boxes[..., 0]) & (boxes[..., 3] > boxes[..., 1])

        return (
            images,
            torch.where(keep.unsqueeze(-1), boxes, gt_boxes.new_zeros(gt_boxes.shape)),
            torch.where(keep, gt_labels, gt_labels.new_full(gt_labels.shape, -1)),
            torch.where(
                keep.view(*keep.shape, 1, 1),
                keypoints,
                gt_keypoints.new_zeros(gt_keypoints.shape),
            ),
        )
