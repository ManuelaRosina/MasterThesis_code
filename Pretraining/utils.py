import torch
from torchvision.ops.boxes import box_area
from typing import Any, Dict, List

class NestedTensor(object):
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        cast_mask = self.mask.to(*args, **kwargs) if self.mask is not None else None
        return type(self)(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    @classmethod
    def from_tensor_list(cls, tensor_list, do_round=False):
        # TODO make this more general
        if tensor_list[0].ndim == 3:
            # TODO make it support different-sized images
            max_size = tuple(max(s) for s in zip(*[img.shape for img in tensor_list]))
            # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
            batch_shape = (len(tensor_list),) + max_size
            b, c, h, w = batch_shape
            if do_round:
                # Round to an even size to avoid rounding issues in fpn
                p = 128
                h = h if h % p == 0 else (h // p + 1) * p
                w = w if w % p == 0 else (w // p + 1) * p
                batch_shape = b, c, h, w

            dtype = tensor_list[0].dtype
            device = tensor_list[0].device
            tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
            for img, pad_img, m in zip(tensor_list, tensor, mask):
                pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
                m[: img.shape[1], : img.shape[2]] = False
        else:
            raise ValueError("not supported")
        return cls(tensor, mask)

    def __repr__(self):
        return repr(self.tensors)

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=b.device)
    return b


def normalize_boxes(out_bbox, image_size):
    # convert to [x0, y0, x1, y1] format
    boxes = box_cxcywh_to_xyxy(out_bbox)
    # and from absolute [0, height] to relative [0, 1] coordinates
    print("size: ", image_size)
    img_h, img_w = image_size, image_size
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes / scale_fct[:, None, :]

    return boxes


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def dist_iou(boxes1, boxes2):
    """
    DIoU from https://arxiv.org/pdf/1911.08287.pdf

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    #print('iou: ', boxes1)
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    boxes1_center = boxes1[:, 2:] - boxes1[:, :2]
    boxes1_center = boxes1[:, 2:] + boxes1_center
    
    boxes2_center = boxes2[:, 2:] - boxes2[:, :2]
    boxes2_center = boxes2[:, 2:] + boxes2_center
    
    print("boxes1: ", boxes1.shape)
    print("boxes2: ", boxes2.shape) 
    print("b1_c: ", boxes1_center.shape)
    print("b2_c: ", boxes2_center.shape)
    print("lt: ", lt.shape) 
    print("rb: ", rb.shape)
    print("iou: ", iou.shape)
    
    dist = torch.cdist(boxes1_center, boxes2_center)
    diag_dist = torch.cdist(lt, rb)
    
    print("dist: ", dist.shape)
    print("diag_dist: ", diag_dist.shape)
    print()

    return 1 - iou + ( dist / diag_dist )


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    #print('iou: ', boxes1)
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]
    
    #print("area: ", area.shape)
    #print("union: ", union.shape)
    #print("iou: ", iou.shape)

    return iou - (area - union) / area


def collate_fn(batch):
    batch = list(zip(*batch))
    final_batch = {}

    tensor_list = batch[0]

    if tensor_list[0].ndim == 3:
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = (len(tensor_list),) + max_size
        b, c, h, w = batch_shape

        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        for img, pad_img in zip(tensor_list, tensor):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    final_batch["samples"] = tensor
    final_batch["targets"] = batch[1]

    return final_batch


def targets_to(targets: List[Dict[str, Any]], device):
    """Moves the target dicts to the given device."""
    included_keys = ["boxes", "scaled boxes"]
    return [{k: v.to(device) if k in included_keys else v for k, v in t.items() if k != "caption"} for t in targets]
