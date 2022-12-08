import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict
import numpy as np
from torchvision.ops.boxes import box_area
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Any, Dict, List, Optional


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


# for output bounding box post-processing

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
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
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
    #print(boxes2)
    #print((boxes2[:, 2:] >= boxes2[:, :2]).all())
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        print('error box: ', boxes2)
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


"""
# TODO: what to do with different image sizes?
def process_model_output(model_outputs, class_names, image_size,  confidence=0.7):
    # keep only predictions with 0.7+ confidence
    probas = model_outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > confidence  # TODO: which confidence?
    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(model_outputs['pred_boxes'][0, keep], image_size)
    scores = probas[keep]

    predictions = []

    for p, (xmin, ymin, xmax, ymax) in zip(scores, bboxes_scaled.tolist()):
        cl = p.argmax()
        predictions.append({'Class': class_names[cl], 'Probability': p[cl].item(), 'Bbox': (xmin, ymin, xmax, ymax)})

    return predictions


def print_image_prediction(image, predictions, figsize=(16, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(image)
    ax = plt.gca()
    for pred in predictions:
        xmin, ymin, xmax, ymax = pred['Bbox']
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, linewidth=3))
        cl = pred['Class']
        prop = pred['Probability']
        text = f'{cl}: {prop:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()
"""

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, cache, target_sizes):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        #prob = 1 - F.softmax(out_logits, -1)[0, :, -1]
        #keep = (prob > 0.7).cpu()
        #scores, labels = prob[..., :-1].max(-1)

        #labels = torch.ones_like(labels)

        #scores = 1 - prob[:, :, -1]

        # convert to [x0, y0, x1, y1] format
        boxes = box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # Extract the text spans predicted by each box
        positive_tokens = (out_logits.softmax(-1) > 0.01).nonzero().tolist()
        #print('logits shape: ', out_logits.shape)
        #print('pos tokes: ', positive_tokens)
        predicted_spans = defaultdict(str)
        seen_tok = defaultdict(set)
        for tok in positive_tokens:
            item = str(tok[0]) + '_' + str(tok[1])
            pos = tok[-1]

            if len(predicted_spans[item]) == 0:
                predicted_spans[item] = ' '

            if pos < len(cache["tokenized"]['input_ids'][0]):
                #span = cache["tokenized"].token_to_chars(tok[0], pos)
                word_nr = cache["tokenized"].token_to_word(tok[0], pos)
                idxs = np.where(np.array(cache["tokenized"][tok[0]].words) == word_nr)[0]
                if idxs[0] not in seen_tok[item]:
                    word = ''.join(cache["tokenized"][tok[0]].tokens[idxs[0]:idxs[-1]]).replace('#','')
                    predicted_spans[item] += " " + word
                    seen_tok[item].update(idxs)

        label = defaultdict(list)
        for i in range(out_bbox.shape[0]):
            label[str(i)] = []
        for key, value in predicted_spans.items():
            sent_nr, _ = key.split('_')
            label[sent_nr].append(value)
        labels = [label[k] for k in list(label.keys())]
        #print(out_bbox.shape)
        #print('predicted_spans: ', predicted_spans)
        #print('labels: ', labels)
        #print('labels: ', len(labels), ' boxes: ', len(boxes))
        #print(boxes)
        assert len(labels) == len(boxes) #assert len(prob) == len(labels) == len(boxes)
        results = [{"labels": l, "boxes": b} for l, b in zip(labels, boxes)]#results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(prob, labels, boxes)]
        #print('results: ', results)
        return results, results


def build_postprocessor():
    postprocessors = PostProcessTest()#PostProcessTest() #PostProcess()
    return postprocessors


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty channel sizes.
    """
    if input.numel() > 0:
        return torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners)

    assert input.shape[0] != 0 or input.shape[1] != 0, "At least one of the two first dimensions must be non zero"

    if input.shape[1] == 0:
        # Pytorch doesn't support null dimension on the channel dimension, so we transpose to fake a null batch dim
        return torch.nn.functional.interpolate(input.transpose(0, 1), size, scale_factor, mode, align_corners).transpose(0, 1)

    # empty batch dimension is now supported in pytorch
    return torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners)



def collate_fn(do_round, batch):
    batch = list(zip(*batch))
    final_batch = {}
    final_batch["samples"] = NestedTensor.from_tensor_list(batch[0], do_round)
    final_batch["targets"] = batch[1]
    if "positive map" in batch[1][0]:
        # we batch the positive maps here
        # Since in general each batch element will have a different number of boxes,
        # we collapse a single batch dimension to avoid padding. This is sufficient for our purposes.
        max_len = max([v["positive map"].shape[1] for v in batch[1]])
        nb_boxes = sum([v["positive map"].shape[0] for v in batch[1]])
        batched_pos_map = torch.zeros((nb_boxes, max_len), dtype=torch.bool)
        cur_count = 0
        for v in batch[1]:
            cur_pos = v["positive map"]
            batched_pos_map[cur_count : cur_count + len(cur_pos), : cur_pos.shape[1]] = cur_pos
            cur_count += len(cur_pos)

        assert cur_count == len(batched_pos_map)
        # assert batched_pos_map.sum().item() == sum([v["positive_map"].sum().item() for v in batch[1]])
        final_batch["positive map"] = batched_pos_map.float()
    if "positive_map_eval" in batch[1][0]:
        # we batch the positive maps here
        # Since in general each batch element will have a different number of boxes,
        # we collapse a single batch dimension to avoid padding. This is sufficient for our purposes.
        max_len = max([v["positive_map_eval"].shape[1] for v in batch[1]])
        nb_boxes = sum([v["positive_map_eval"].shape[0] for v in batch[1]])
        batched_pos_map = torch.zeros((nb_boxes, max_len), dtype=torch.bool)
        cur_count = 0
        for v in batch[1]:
            cur_pos = v["positive_map_eval"]
            batched_pos_map[cur_count : cur_count + len(cur_pos), : cur_pos.shape[1]] = cur_pos
            cur_count += len(cur_pos)

        assert cur_count == len(batched_pos_map)
        # assert batched_pos_map.sum().item() == sum([v["positive_map"].sum().item() for v in batch[1]])
        final_batch["positive_map_eval"] = batched_pos_map.float()
    if "answer" in batch[1][0] or "answer_type" in batch[1][0]:
        answers = {}
        for f in batch[1][0].keys():
            if "answer" not in f:
                continue
            answers[f] = torch.stack([b[f] for b in batch[1]])
        final_batch["answers"] = answers

    return final_batch


def collate_fn_deep_lesion(batch):
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
    

def print_image_prediction(image, predictions, figsize=(16, 10)):
    #plt.figure(figsize=figsize)
    #plt.imshow(image, cmap='gray', aspect='auto')
    #ax = plt.gca()
    fig, ax = plt.subplots()
    #print(image.shape)
    ax.imshow(image[:, :, 1], cmap='gray', aspect='auto')
    for i, box in enumerate(predictions['boxes']):
        cl = predictions['labels'][i]
        if cl:
            text = f'{cl}'
            xmin, ymin, xmax, ymax = box
            ax.add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, linewidth=3))
            ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


def print_image_prediction_2(image, predictions, figsize=(16, 10)):
    #plt.figure(figsize=figsize)
    #plt.imshow(image, cmap='gray', aspect='auto')
    #ax = plt.gca()
    fig, ax = plt.subplots()
    #print(image.shape)
    ax.imshow(image, aspect='auto')
    for i, box in enumerate(predictions['boxes']):
        cl = predictions['labels'][i]
        if cl:
            text = f'{cl}'
            xmin, ymin, xmax, ymax = box
            ax.add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, linewidth=3))
            ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


class PostProcessTest(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, score_thresh=0.7):
        super().__init__()
        self.score_thresh = score_thresh

    @torch.no_grad()
    def forward(self, outputs, cache, target_sizes):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)
        #labels = torch.ones_like(labels)

        scores = 1 - scores
        #scores = prob[:, :, -1]

        # convert to [x0, y0, x1, y1] format
        boxes = box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]

        assert len(scores) == len(labels) == len(boxes)
        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]
        #print("max score: ", scores.max(-1))
        final_results = []
        for i, elem in enumerate(results):
            #print('scores: ', elem["scores"])
            keep = elem["scores"] > self.score_thresh
            # convert boxes to [x0, y0, w, h]
            boxes = elem["boxes"][keep].view(-1, 4)
            boxes[..., 2:] -= boxes[..., :2]

            positive_tokens = (outputs["pred_logits"][i, keep].softmax(-1) > 0.08).nonzero().tolist()
            #print(outputs["pred_logits"][i, keep].softmax(-1).max(dim=1))
            #print('pos tok: ', outputs["pred_logits"][i, keep].softmax(-1) > 0.01)
            predicted_spans = defaultdict(str)
            for item in range(outputs["pred_logits"][i].shape[0]):
                predicted_spans[item] = ''
            for tok in positive_tokens:
                item, pos = tok
                if pos < 255:
                    try:
                        span = cache["tokenized"].token_to_chars(0, pos)
                        predicted_spans[item] += " " + cache['text'][i][span.start:span.end]
                    except:
                        continue

            labels = [predicted_spans[k] if predicted_spans[k] != '' else None for k in sorted(list(predicted_spans.keys()))]

            if not labels:
                labels = [None for _ in range(len(keep))]


            res = {"scores": elem["scores"][keep], "labels": labels, "boxes": boxes.tolist()}
            final_results.append(res)

        return results, final_results
        
        
def targets_to(targets: List[Dict[str, Any]], device):
    """Moves the target dicts to the given device."""
    excluded_keys = [
        "questionId",
        "tokens_positive",
        "tokens",
        "dataset_name",
        "sentence_id",
        "original_img_id",
        "nb_eval",
        "task_id",
        "original_id",
        "image_id",
        "sentence",
        "labels",
    ]
    included_keys = ["boxes", "scaled boxes"]
    return [{k: v.to(device) if k in included_keys else v for k, v in t.items() if k != "caption"} for t in targets]

