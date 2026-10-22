import torch


def make_anchors(
    h: int, w: int, device: str | torch.device, dtype: torch.dtype, offset: float = 0.5
) -> torch.Tensor:
    """Generate anchor points for a grid of size (h, w).

    Parameters
    ----------
    h : int
        Height of the grid.
    w : int
        Width of the grid.
    device : str | torch.device
        Device to place the anchor points on.
    dtype : torch.dtype
        Data type of the anchor points.
    offset : float, optional
        Offset for the anchor points, by default 0.5.

    Returns
    -------
    torch.Tensor
        A tensor of shape (h * w, 2) containing the anchor points in the shape [H, W, 2] where the
        last dimension represents (x, y) coordinates.
    """
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype) + offset,
        torch.arange(w, device=device, dtype=dtype) + offset,
        indexing="ij",
    )
    anchors = torch.stack((grid_x, grid_y), dim=-1)
    return anchors.view(-1, 2)
