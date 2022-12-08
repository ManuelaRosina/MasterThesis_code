from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from prettytable import PrettyTable


#### Bounding box utilities imported from torchvision and converted to numpy
def box_area(boxes: np.array) -> np.array:
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        area (Tensor[N]): area for each box
    """
    assert boxes.ndim == 2 and boxes.shape[-1] == 4
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def _box_inter_union(boxes1: np.array, boxes2: np.array) -> Tuple[np.array, np.array]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union


def box_iou(boxes1: np.array, boxes2: np.array) -> np.array:
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou

#### End of import of box utilities


class RecallTracker:
    """ Utility class to track recall@k for various k, split by categories"""

    def __init__(self, topk: Sequence[int]):
        """
        Parameters:
           - topk : tuple of ints corresponding to the recalls being tracked (eg, recall@1, recall@10, ...)
        """

        self.total_byk_bycat: Dict[int, Dict[str, int]] = {k: defaultdict(int) for k in topk}
        self.positives_byk_bycat: Dict[int, Dict[str, int]] = {k: defaultdict(int) for k in topk}

    def add_positive(self, k: int, category: str):
        """Log a positive hit @k for given category"""
        if k not in self.total_byk_bycat:
            raise RuntimeError(f"{k} is not a valid recall threshold")
        self.total_byk_bycat[k][category] += 1
        self.positives_byk_bycat[k][category] += 1

    def add_negative(self, k: int, category: str):
        """Log a negative hit @k for given category"""
        if k not in self.total_byk_bycat:
            raise RuntimeError(f"{k} is not a valid recall threshold")
        self.total_byk_bycat[k][category] += 1

    def report(self) -> Dict[int, Dict[str, float]]:
        """Return a condensed report of the results as a dict of dict.
        report[k][cat] is the recall@k for the given category
        """
        report: Dict[int, Dict[str, float]] = {}
        for k in self.total_byk_bycat:
            assert k in self.positives_byk_bycat
            report[k] = {
                cat: self.positives_byk_bycat[k][cat] / self.total_byk_bycat[k][cat] for cat in self.total_byk_bycat[k]
            }
        return report


class RecallEvaluator:
    def __init__(
        self,
        topk: Sequence[int] = (1, 5, 10, -1),
        iou_thresh: float = 0.5,
    ):

        self.topk = topk
        self.iou_thresh = iou_thresh

    def evaluate(self, predictions: List[Dict], targets):
        evaluated_ids = set()

        recall_tracker = RecallTracker(self.topk)

        for pred, target in zip(predictions, targets):
            cur_id = f"{target['image_id']}_{target['sentence_id']}"
            if cur_id in evaluated_ids:
                print(
                    "Warning, multiple predictions found for sentence"
                    f"{target['sentence_id']} in image {target['image_id']}"
                )
                continue
            evaluated_ids.add(cur_id)
            pred_boxes = pred["boxes"].to("cpu")
            target_boxes = target['boxes'].to("cpu")

            for cur_boxes in pred_boxes:
                ious = box_iou(np.expand_dims(np.asarray(cur_boxes), axis=0), np.asarray(target_boxes))
                for k in self.topk:
                    if k == -1:
                        maxi = ious.max()
                    else:
                        assert k > 0
                        maxi = ious[:k].max()
                    if maxi >= self.iou_thresh:
                        recall_tracker.add_positive(k, "all")
                    else:
                        recall_tracker.add_negative(k, "all")

        return recall_tracker.report()


class Evaluator(object):
    def __init__(
        self,
        top_k=(1, 5, 10, -1),
        iou_thresh=0.5,
    ):
        assert isinstance(top_k, (list, tuple))

        self.evaluator = RecallEvaluator(
            topk=top_k, iou_thresh=iou_thresh
        )
        self.predictions = []
        self.targets = []
        self.results = None

    def update(self, predictions, targets):
        self.predictions += predictions
        self.targets += targets

    def summarize(self):
        self.results = self.evaluator.evaluate(self.predictions, self.targets)
        table = PrettyTable()
        all_cat = sorted(list(self.results.values())[0].keys())
        table.field_names = ["Recall@k"] + all_cat

        score = {}
        for k, v in self.results.items():
            cur_results = [v[cat] for cat in all_cat]
            header = "Upper_bound" if k == -1 else f"Recall@{k}"

            for cat in all_cat:
                score[f"{header}_{cat}"] = v[cat]
            table.add_row([header] + cur_results)

        #print(table)

        return score
