# build the complete model pipeline
import torch
from torch import nn, tensor
import torch.nn.functional as F
from transformer import build_transformer
from backbone import build_backbone
import utils
from utils import NestedTensor, normalize_boxes
from matcher import build_matcher
import dist
from collections import OrderedDict

def build_model(args):
    num_classes = 255 # TODO: not hardcode?
    device = torch.device(args.device)
    back = build_backbone(args)
    transformer = build_transformer(args)
    model = MDETR(
        back,
        transformer,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        contrastive_hdim=args.contrastive_loss_hdim,
        contrastive_align_loss=args.contrastive_align_loss,
    )

    if args.pretrained:
        # TODO: need to be local for cluster
        #checkpoint = torch.hub.load_state_dict_from_url(
        #    url="https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth",
        #    map_location="cpu",
        #    check_hash=True,
        #)
        checkpoint = torch.load(args.path_chansey + args.model_checkpoint)#, map_location=torch.device('cpu'))
        new_check = OrderedDict()
        if args.model_checkpoint == 'pretrained_Mdetr_resnet101_checkpoint.pth':
            for key, value in checkpoint["model"].items():
                #if "body" in key:
                    #"backbone.0.0.body.conv1.weight" <- "backbone.0.body.conv1.weight"
                    #new_check[key.replace("backbone.0", "backbone.0.0")] = value
                    #print("changes: ", key)
                if "resizer" in key:
                    #backbone.1.resizer.fc.weight <- transformer.resizer.fc.weight
                    new_check[key.replace('transformer', 'backbone.1')] = value
                    #print("changes: ", key)
                else:
                    new_check[key] = value
            model.load_state_dict(new_check, strict=False)
        else:
             # remove text backbone weights
            new_check = OrderedDict()
            for key, value in checkpoint["model_state_dict"].items():
                if "backbone.1.text_encoder" not in key:
                    new_check[key] = value
            model.load_state_dict(new_check, strict=False)
            #model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            
        print("loaded checkpoint: ", args.model_checkpoint)

    matcher = build_matcher(args)

    weight_dict = {"loss_bbox": args.bbox_loss_coef}
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef
    if args.ce_loss_coef != 0:
        weight_dict["loss_ce"] = args.ce_loss_coef
    weight_dict["loss_giou"] = args.giou_loss_coef

    losses = ["boxes", "cardinality"]
    if args.ce_loss_coef != 0:
        losses += ["labels"]
    if args.contrastive_align_loss:
        losses += ["contrastive_align"]

    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        eos_coef=args.eos_coef,
        losses=losses,
        temperature=args.temperature_NCE,
    )
    criterion.to(device)

    return model, criterion, weight_dict

class MDETR(nn.Module):
    """ This is the MDETR module that performs modulated object detection """

    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        aux_loss=False,
        contrastive_hdim=64,
        contrastive_align_loss=False,
    ):
        """Initializes the model.

        Args:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         MDETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            contrastive_hdim: dimension used for projecting the embeddings before computing contrastive loss
            contrastive_align_loss: If true, perform box - token contrastive learning
        """
        super().__init__()

        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.backbone = backbone
        self.aux_loss = aux_loss

        self.contrastive_align_loss = contrastive_align_loss
        if contrastive_align_loss:
            self.contrastive_align_projection_image = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_text = nn.Linear(hidden_dim, contrastive_hdim)

    def forward(self, image, text):
        """The forward expects a NestedTensor, which consists of:
                   - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
                   - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

                It returns a dict with the following elements:
                   - "pred_logits": the classification logits (including no-object) for all queries.
                                    Shape= [batch_size x num_queries x (num_classes + 1)]
                   - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                                   (center_x, center_y, height, width). These values are normalized in [0, 1],
                                   relative to the size of each individual image (disregarding possible padding).
                                   See PostProcess for information on how to retrieve the unnormalized bounding box.
                   - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                    dictionnaries containing the two above keys for each decoder layer.
                """
        if not isinstance(image, NestedTensor):
            image = NestedTensor.from_tensor_list(image)
        #print('model input: ', text)
        src, mask, pos_embed, text_backbone_memory = self.backbone(image, text)
        #print('text_backbone_memory: ', text_backbone_memory)

        hw, bs, c = src.shape
        #print("src shape ", src.shape)
        query_embed = self.query_embed.weight
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        #print("query emb ", query_embed.shape)
        hs, memory_cache = self.transformer(
            src,
            mask=mask,
            query_embed=query_embed,
            pos_embed=pos_embed,
            text_backbone_memory=text_backbone_memory
        )

        memory_cache['text'] = text

        out = {}
        #print("hs shape", hs)
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        #print("outputs_class ", outputs_class.shape) # torch.Size([6, 1, 3, 256])
        #print("outputs_coord ", outputs_coord) # torch.Size([6, 1, 3, 4])

        out.update(
            {
                "pred_logits": outputs_class[-1],
                "pred_boxes": outputs_coord[-1],
            }
        )

        proj_queries, proj_tokens = None, None
        if self.contrastive_align_loss:
            proj_queries = F.normalize(self.contrastive_align_projection_image(hs), p=2, dim=-1)
            proj_tokens = F.normalize(
                self.contrastive_align_projection_text(memory_cache["text_memory"]).transpose(0, 1), p=2, dim=-1
            )
            out.update(
                {
                    "proj_queries": proj_queries[-1],
                    "proj_tokens": proj_tokens,
                    "tokenized": memory_cache["tokenized"],
                }
            )
        if self.aux_loss:
            if self.contrastive_align_loss:
                assert proj_tokens is not None and proj_queries is not None
                out["aux_outputs"] = [
                    {
                        "pred_logits": a,
                        "pred_boxes": b,
                        "proj_queries": c,
                        "proj_tokens": proj_tokens,
                        "tokenized": memory_cache["tokenized"],
                    }
                    for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], proj_queries[:-1])
                ]
            else:
                out["aux_outputs"] = [
                    {
                        "pred_logits": a,
                        "pred_boxes": b,
                    }
                    for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
                ]

        return out, memory_cache


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        for i, layer in enumerate(self.layers):
            torch.nn.init.orthogonal_(layer.weight)
        #    torch.nn.init.xavier_uniform(layer.weight)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, eos_coef, losses, temperature):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses
        self.temperature = temperature
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, positive_map, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """

        logits = outputs["pred_logits"].log_softmax(-1)  # BS x (num_queries) x (num_tokens)

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = []
        offset = 0
        for i, (_, tgt) in enumerate(indices):
            tgt_idx.append(tgt + offset)
            offset += len(targets[i]["boxes"])
        tgt_idx = torch.cat(tgt_idx)

        tgt_pos = positive_map[tgt_idx]
        target_sim = torch.zeros_like(logits)
        target_sim[:, :, -1] = 1
        target_sim[src_idx] = tgt_pos

        loss_ce = -(logits * target_sim).sum(-1)

        eos_coef = torch.full(loss_ce.shape, self.eos_coef, device=target_sim.device)
        eos_coef[src_idx] = 1

        loss_ce = loss_ce * eos_coef
        loss_ce = loss_ce.sum() / num_boxes

        losses = {"loss_ce": loss_ce}

        return losses

    def loss_contrastive_align(self, outputs, targets, positive_map, indices, num_boxes):
        bs = outputs["proj_queries"].shape[0]
        tokenized = outputs["tokenized"]

        normalized_text_emb = outputs["proj_tokens"]  # BS x (num_tokens) x hdim
        normalized_img_emb = outputs["proj_queries"]  # BS x (num_queries) x hdim

        logits = (
            torch.matmul(normalized_img_emb, normalized_text_emb.transpose(-1, -2)) / self.temperature
        )  # BS x (num_queries) x (num_tokens)

        # construct a map such that positive_map[k, i,j] = True iff query i is associated to token j in batch item k
        # For efficency, the construction happens on CPU, then the whole matrix is transferred to GPU in one go.
        positive_map = torch.zeros(logits.shape, dtype=torch.bool)
        for i, ((idx_src, idx_tgt), tgt) in enumerate(zip(indices, targets)):
            if "tokens_positive" in tgt:
                cur_tokens = [tgt["tokens_positive"][j] for j in idx_tgt] # cur_tokens: a list of tokens associated with target
            else:
                cur_tokens = [tgt["target"][j] for j in idx_tgt]

            for j, tok_list in enumerate(cur_tokens):
                for (beg, end) in tok_list:
                    beg_pos = tokenized.char_to_token(i, beg)
                    end_pos = tokenized.char_to_token(i, end - 1)
                    if beg_pos is None:
                        try:
                            beg_pos = tokenized.char_to_token(beg + 1)
                            if beg_pos is None:
                                beg_pos = tokenized.char_to_token(beg + 2)
                        except:
                            beg_pos = None
                    if end_pos is None:
                        try:
                            end_pos = tokenized.char_to_token(end - 2)
                            if end_pos is None:
                                end_pos = tokenized.char_to_token(end - 3)
                        except:
                            end_pos = None
                    if beg_pos is None or end_pos is None:
                        continue

                    assert beg_pos is not None and end_pos is not None
                    #positive_map[batch_index, bbox, token position start : end]
                    positive_map[i, idx_src[j], beg_pos : end_pos + 1].fill_(True)

        positive_map = positive_map.to(logits.device)
        positive_logits = -logits.masked_fill(~positive_map, 0)
        negative_logits = logits  # .masked_fill(positive_map, -1000000)

        boxes_with_pos = positive_map.any(2)
        pos_term = positive_logits.sum(2)
        neg_term = negative_logits.logsumexp(2)

        nb_pos = positive_map.sum(2) + 1e-6 # why add 1e-6?

        box_to_token_loss = ((pos_term / nb_pos + neg_term)).masked_fill(~boxes_with_pos, 0).sum()

        tokens_with_pos = positive_map.any(1)
        pos_term = positive_logits.sum(1)
        neg_term = negative_logits.logsumexp(1)

        nb_pos = positive_map.sum(1) + 1e-6

        tokens_to_boxes_loss = ((pos_term / nb_pos + neg_term)).masked_fill(~tokens_with_pos, 0).sum()
        tot_loss = (box_to_token_loss + tokens_to_boxes_loss) / 2

        return {"loss_contrastive_align": tot_loss.sum()/ num_boxes}

    def loss_boxes(self, outputs, targets, positive_map, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        #target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_boxes = torch.cat([t["scaled boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        #loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="sum")

        losses = {}
        losses["loss_bbox"] = loss_bbox / num_boxes

        loss_giou = 1 - torch.diag(
            utils.generalized_box_iou(utils.box_cxcywh_to_xyxy(src_boxes), utils.box_cxcywh_to_xyxy(target_boxes))
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, positive_map, indices, num_boxes):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["boxes"]) for v in targets], device=device)
        ## Count the number of predictions that are NOT "no-object" (which is the last class)
        # normalized_text_emb = outputs["proj_tokens"]  # BS x (num_tokens) x hdim
        # normalized_img_emb = outputs["proj_queries"]  # BS x (num_queries) x hdim

        # logits = torch.matmul(
        #    normalized_img_emb, normalized_text_emb.transpose(-1, -2)
        # )  # BS x (num_queries) x (num_tokens)
        # card_pred = (logits[:, :, 0] > 0.5).sum(1)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
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

    def get_loss(self, loss, outputs, targets, positive_map, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "contrastive_align": self.loss_contrastive_align,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, positive_map, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, positive_map=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        #print(outputs["pred_boxes"])
        indices = self.matcher(outputs_without_aux, targets, positive_map)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes / 1, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, positive_map, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets, positive_map)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, positive_map, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
