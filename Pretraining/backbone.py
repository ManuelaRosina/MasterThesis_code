from torchvision.models import vgg16, vgg16_bn, resnet101, resnet18, resnet34, resnet50
import torch
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from collections import OrderedDict
import torch.nn.functional as F
import utils
from matcher import build_matcher


def build_backbone(args):
    backbone = Backbone(args.num_channels, args.hidden_dim, args.img_back_pretrained, args.backbone_type)
    matcher = build_matcher(args)
    weight_dict = {}
    weight_dict["loss_bbox"] = args.bbox_loss_coef
    weight_dict["loss_giou"] = args.giou_loss_coef
    weight_dict["mse"] = args.mse_loss_coef
    losses = ["boxes"]
    criterion = SetCriterion(
        matcher=matcher,
        losses=losses,
    )

    return backbone, criterion, weight_dict


class Backbone(nn.Module):
    def __init__(self, num_channels=2048, hidden_dim=256, pretrained=False, net_type='vgg'):
        super().__init__()
        self.num_channels = num_channels
        
        if net_type == 'vgg':
            self.backbone = vgg16(pretrained=pretrained)
            del self.backbone.classifier

        if 'resnet' in net_type:
            if net_type == 'resnet18':
                self.backbone = resnet18(pretrained=pretrained)
            elif net_type == 'resnet34':
                self.backbone = resnet34(pretrained=pretrained)
            elif net_type == 'resnet50':
                self.backbone = resnet50(pretrained=pretrained)
            elif net_type == 'resnet101':
                self.backbone = resnet101(pretrained=pretrained)
            else:
                print("ERROR")
            del self.backbone.fc
            
        #self.backbone = resnet50(pretrained=True)#, norm_layer=FrozenBatchNorm2d)
        #self.backbone = resnet101(pretrained=True)#, norm_layer=FrozenBatchNorm2d)
        #del self.backbone.fc
        #self.backbone = vgg16()
        #del self.backbone.classifier

        # create conversion layer
        if net_type == 'resnet50' or net_type == 'resnet101':
            self.conv = nn.Conv2d(2048, self.num_channels, 1)
        else:
            self.conv = nn.Conv2d(512, self.num_channels, 1)
        
        #del self.backbone.fc

        # create conversion layer
        #self.conv = nn.Conv2d(2048, self.num_channels, 1)
        #self.conv = nn.Conv2d(512, self.num_channels, 1)
        #print(self.num_channels)

        # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        # return_layers = {"layer4": "0"}
        return_layers = {"avgpool": "0"}
        self.body = IntermediateLayerGetter(self.backbone, return_layers=return_layers)
        
        self.flat = nn.Flatten()
        #self.linear1 = nn.Linear(25088, 128)
        self.linear1 = nn.Linear(self.num_channels, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, 8)

    def forward(self, inputs):
        xs = self.body(inputs)
        out = []
        for name, x in xs.items():
            #print(x.shape)
            x = self.conv(x)
            #print(x.shape)
            x = self.flat(x)
            #print(x.shape)
            x = self.linear1(x).relu()
            x = self.linear2(x).relu()
            x = self.linear3(x).relu()
            x = self.linear4(x).sigmoid()

            out.append(x.reshape((-1, 2, 4)))

        return out[-1]


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, losses):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = matcher
        self.losses = losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        #idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"].flatten(0,1)#[idx]
        #target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        #target_boxes = torch.cat([t["scaled boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_boxes = torch.cat([t["scaled boxes"] for t in targets], dim=0)
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="sum")

        #print("src_boxes: ")
        #print(src_boxes)
        #print("target_boxes: ")
        #print(target_boxes)

        losses = {}
        losses["loss_bbox"] = loss_bbox #/ num_boxes

        loss_giou = 1 - torch.diag(
            utils.generalized_box_iou(utils.box_cxcywh_to_xyxy(src_boxes), utils.box_cxcywh_to_xyxy(target_boxes))
        )
        losses["loss_giou"] = loss_giou.sum() #/ num_boxes

        losses["mse"] = F.mse_loss(src_boxes, target_boxes, reduction="sum") / num_boxes

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # Retrieve the matching between the outputs of the last layer and the targets
        #print(outputs["pred_boxes"])
        indices = None #self.matcher(outputs, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes / 1, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        return losses
