import torch

type XYXYBox = torch.Tensor  # [x1, y1, x2, y2]
type XYWHBox = torch.Tensor  # [x, y, w, h]


def bbox_iou(box1: XYXYBox, box2: XYXYBox, eps: float = 1e-7, c_iou: bool = True) -> torch.Tensor:
    """
    Compute the Intersection over Union (IoU) or Complete IoU (CIoU) of two sets of boxes.

    Parameters
    ----------
    box1 : XYXYBox
        A tensor of shape (N, 4) representing N boxes in the format [x1, y1, x2, y2].
    box2 : XYXYBox
        A tensor of shape (M, 4) representing M boxes in the format [x1, y1, x2, y2].
    eps : float
        A small value to avoid division by zero.
    c_iou : bool
        If True, calculates Complete IoU (CIoU). If False, calculates standard IoU.

    Returns
    -------
    torch.Tensor
        A tensor of shape (N, M) containing the IoU or CIoU values between each pair of boxes.
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0:1], box1[:, 1:2], box1[:, 2:3], box1[:, 3:4]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[None, :, 0], box2[None, :, 1], box2[None, :, 2], box2[None, :, 3]

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    area1 = w1 * h1
    area2 = w2 * h2

    # Union area and standard IoU -> shape (N, M)
    union_area = area1 + area2 - inter_area + eps
    iou = inter_area / union_area

    if not c_iou:
        return iou

    # Enclosing box (convex hull) dimensions for distance penalty
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps  # Enclosing diagonal squared

    # Center points distance squared
    b1_cx, b1_cy = (b1_x1 + b1_x2) / 2.0, (b1_y1 + b1_y2) / 2.0
    b2_cx, b2_cy = (b2_x1 + b2_x2) / 2.0, (b2_y1 + b2_y2) / 2.0
    rho2 = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2

    # Aspect ratio consistency term v and weight parameter alpha
    v = (4.0 / (torch.pi ** 2)) * torch.pow(torch.atan(w1 / (h1 + eps)) - torch.atan(w2 / (h2 + eps)), 2)

    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)

    # Complete IoU (CIoU) formula
    ciou = iou - (rho2 / c2 + alpha * v)
    return ciou