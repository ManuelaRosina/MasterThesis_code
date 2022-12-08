from model import MDETR
from backbone import build_position_encoding, Backbone, Joiner, TextBackbone, Concat
from transformer import build_transformer
from argparse import Namespace
from torch import nn
import torch
from transformers import RobertaModel, RobertaTokenizerFast, BertModel, BertTokenizerFast
from utils import build_postprocessor, generalized_box_iou, collate_fn
import utils
import prepare_dataset
import time
import datetime
from torch.utils.data import DataLoader
from collections import OrderedDict

def center_distcance(pred, tgt):
    #if len(tgt) > 2:
    #    tgt = tgt[:2,:]
    tgt_center = tgt[:, 2:] - tgt[:, :2]
    tgt_center = tgt[:, 2:] + tgt_center
    
    pred_center = pred[:, 2:] - pred[:, :2] #pred[:len(tgt), 2:] - pred[:len(tgt), :2]
    pred_center = pred[:, 2:] + pred_center
    
    #print(tgt)
    #print(tgt_center.shape)
    #print(pred)
    #print(pred_center)
    #print(tgt_center)
    lesion_length = torch.nn.functional.pairwise_distance(tgt[:, 1], tgt[:, 3])
    #print("ll:", lesion_length)
    
    dist = torch.cdist(tgt_center, pred_center) #torch.nn.functional.pairwise_distance(tgt_center, pred_center)
    #print("matrix: ", dist)
    #print(lesion_length)
    #print()
    #print("dist:", dist)
    #print(dist/lesion_length)
    rel = torch.log(dist/(lesion_length*0.5) + 1) #dist/(lesion_length*0.5)
    values, idx = dist.min(axis=1)
    
    return rel, values, idx
    
def evaluate2(pred, tgt, giou_thresh=0.5):
    correct_pred, total_pred = 0, 0
    tp, fp, fn = 0, 0, 0

    # calculate giou for pred with match ground truth
    giou = utils.generalized_box_iou(pred, tgt) #The boxes should be in [x0, y0, x1, y1] format
    #Returns a[N, M] pairwise matrix, where N = len(pred) and M = len(tgt)
    values, idx = giou.max(axis=1)
    hits = (values >= giou_thresh)
    
    # accuracy
    if int(torch.sum((hits == True))) > 0:
        correct_pred = int(torch.sum((hits == True)))
    total_pred = len(tgt)
    
    #if len(tgt) > 1:
    # precision
    tp = int(torch.sum((hits == True)))
    fn = len(tgt) - tp #len(idx.unique()) # idx are the indexes of ground truth boxes matched with prediction.
    # If one (or more) is missing a ground truth box was not predicted
    #fp = fn # if there are two pred matched to the same tgt there is automatically a fp
    #fp = len(tgt)
        
    return tp, fp, fn, correct_pred, total_pred, giou
    
def dist_evaluate(pred, tgt, dist_thresh=10.0):
    correct_pred, total_pred = 0, 0
    tp, fp, fn = 0, 0, 0

    # calculate giou for pred with match ground truth
    dist, values, idx = center_distcance(pred, tgt)
    dist = dist.gather(1,idx.unsqueeze(1))
    hits = (dist <= dist_thresh)
    
    # accuracy
    if int(torch.sum((hits == True))) > 0:
        correct_pred = int(torch.sum((hits == True)))
    total_pred = len(tgt)
    
    #if len(tgt) > 1:
    # precision
    tp = int(torch.sum((hits == True)))
    fn = len(tgt) - tp #len(idx.unique()) # idx are the indexes of ground truth boxes matched with prediction.
    # If one (or more) is missing a ground truth box was not predicted
    #    fp = fn # if there are two pred matched to the same tgt there is automatically a fp
    #fp = len(tgt)
        
    return tp, fp, fn, correct_pred, total_pred, dist[:len(tgt)]
    

def precision_recall(tp, fp, fn):
    # precision and recall computation
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall

def build_backbone(args):
    # device = args.device
    position_embedding = build_position_encoding(args)

    backbone = Backbone(args.num_channels, args.hidden_dim, args.backbone_checkpoint, args.freeze_image_encoder,
                        args.img_back_pretrained, args.backbone_type)

    input_proj = nn.Conv2d(backbone.num_channels, args.hidden_dim, kernel_size=1)
    cnn_model = Joiner(backbone, position_embedding, input_proj)
    cnn_model.num_channels = backbone.num_channels
    text_model = TextBackbone(text_encoder_type=args.text_encoder_type, freeze_text_encoder=args.freeze_text_encoder,
                              d_model=args.hidden_dim, path_chansey=args.path_chansey, device=args.device)

    model = Concat(cnn_model, text_model)
    model.config = text_model.config

    return model

def build_model(args):
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

    return model


# return number of tp, fp and fn based on giou
# if giou > 0.5 with ground truth count as hit
def evaluate(pred, tgt, giou_thresh=0.5):
    # calculate giou for pred with match ground truth
    giou = generalized_box_iou(pred, tgt)  # The boxes should be in [x0, y0, x1, y1] format
    # Returns a[N, M] pairwise matrix, where N = len(pred) and M = len(tgt)
    # print(giou)
    values, idx = giou.max(axis=1)
    # print(idx.unique())
    hits = (values >= giou_thresh)[:len(tgt)]
    # print(hits)
    tp = int(torch.sum((hits == True)))
    fp = len(hits) - len(idx.unique())  # idx are the indexes of ground truth boxes matched with prediction.
    # If one (or more) is missing a ground truth box was not predicted
    fn = int(len(tgt) - tp)

    if len(tgt) >= 3:
        print("three target boxes")

    return tp, fp, fn, values[:len(tgt)]

def eval_checkpoint(args, giou_thresh=0.5, dist_thresh=1.0):
    device = args.device
    model = build_model(args)
    model.to(device)
    checkpoint = torch.load(args.path_chansey+args.checkpoint) #, map_location=torch.device('cpu'))
    if args.model_mode == 'detr':
        # remove text backbone weights
        new_check = OrderedDict()
        for key, value in checkpoint["model_state_dict"].items():
            if "backbone.1.text_encoder" not in key:
                new_check[key] = value
        model.load_state_dict(new_check, strict=False)
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if args.model_mode == 'mdetr' or args.model_mode == 'detr_rm':
        print("report matched dataset")
        if args.text_encoder_type == 'roberta-base':
            tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        elif args.text_encoder_type == 'biobert':
            tokenizer = BertTokenizerFast.from_pretrained('giacomomiolo/biobert_reupload')
        elif args.text_encoder_type == 'robbert':
            tokenizer = RobertaTokenizerFast.from_pretrained('pdelobelle/robbert-v2-dutch-base')
        elif args.text_encoder_type == 'bertje':
            tokenizer = BertTokenizerFast.from_pretrained('GroNLP/bert-base-dutch-cased')
            
        test_dataset = prepare_dataset.LesionData(args.path_chansey + 'Data/FromReports/', args.path_chansey, tokenizer, split='test')
    elif args.model_mode == 'detr_dl':
        print("DeepLesion dataset")
        test_dataset = prepare_dataset.DeepLesion(args.path_chansey + 'Pretraining/', args.path_chansey + 'Data/volumes/', split='test')
    elif args.model_mode == 'detr_umc':
        print("umc unmatched dataset")
        test_dataset = prepare_dataset.UnmatchedLesionData(args.path_chansey + 'Data/UnmatchedGSPS/', args.path_chansey, split='test')
    

    print("evaluating checkpoint ", args.checkpoint)

    total_tp, total_fp, total_fn = 0, 0, 0
    gious = []
    dists = []
    total_correct_pred, total_total_pred = 0, 0
    d_total_tp, d_total_fp, d_total_fn = 0, 0, 0
    d_total_correct_pred, d_total_total_pred = 0, 0

    print("start evaluation")
    start_time = time.time()
    print("using giou thresh: ", giou_thresh, " and dist thresh: ", dist_thresh)

    for i, (image, targets) in enumerate(test_dataset):
        # print(i)
        if i % 100 == 0:
            print(i)

        samples = image.to(device)
        if args.model_mode == 'detr_rm':
            captions = [""]
        else:
            captions = [targets['sentence']] if 'sentence' in targets else [""]
        tgt = targets['boxes']
        orig_size = targets['orig_size']
	
        #print(samples.unsqueeze(0).shape)
        outputs, cache = model(samples.unsqueeze(0).to(device), captions)
        prediction = utils.rescale_bboxes(outputs["pred_boxes"].to('cpu'), orig_size).squeeze()

        #tp, fp, fn, giou_values = evaluate(prediction, tgt, giou_thresh=giou_thresh)
        #total_tp += tp
        #total_fp += fp
        #total_fn += fn
        #gious.extend(giou_values.tolist())
        
        tp, fp, fn, correct_pred, total_pred, giou = evaluate2(prediction, tgt, giou_thresh=giou_thresh)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_correct_pred += correct_pred
        total_total_pred += total_pred
        #gious.extend(giou.tolist())
        
        tp, fp, fn, correct_pred, total_pred, dist = dist_evaluate(prediction, tgt, dist_thresh=dist_thresh)
        d_total_tp += tp
        d_total_fp += fp
        d_total_fn += fn
        d_total_correct_pred += correct_pred
        d_total_total_pred += total_pred
        dists.extend(dist.tolist())


    print('correct_pred: ', total_correct_pred, ' total_pred: ', total_total_pred)
    print('accuracy: ', total_correct_pred/total_total_pred)
    print('sensitivity: ', total_tp/(total_tp + total_fn))
    print("distance based:")
    print('correct_pred: ', d_total_correct_pred, ' total_pred: ', d_total_total_pred)
    print('accuracy: ', d_total_correct_pred/d_total_total_pred)
    print('sensitivity: ', d_total_tp/(d_total_tp + d_total_fn))
    #print("d_tp: ", total_d_tp, ' d_fp: ', total_d_fp, ' d_fn: ', total_d_fn)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Computation time {}".format(total_time_str))
    #idxs.sort()
    #idxs2.sort()
    #print("giou = ", gious)
    #print()
    #print("dist = ", dists)
    print()
    #print(idxs2)


if __name__ == "__main__":
    args_dict = {'hidden_dim': 256,
                     'num_channels': 512,
                     'num_classes': 255,
                     'num_queries': 3,
                     'text_encoder_type': "bertje",  # 'giacomomiolo/biobert_reupload'
                     'backbone_type': 'resnet101',
                     'img_back_pretrained': False,
                     'freeze_text_encoder': False,  # True,
                     'freeze_image_encoder': False,
                     'dropout': 0.001,
                     'nheads': 8,
                     'dim_feedforward': 2048,
                     'enc_layers': 2,
                     'dec_layers': 2,
                     'pre_norm': False,
                     'pass_pos_and_query': True,
                     'aux_loss': False,
                     'contrastive_loss_hdim': 64,
                     'contrastive_align_loss': False,
                     'pretrained': False, #True, # TODO: change to local
                     'path_chansey': "/output/",
                     'device': "cuda",
                     'backbone_checkpoint': '',
                     'checkpoint': 'Final_Detr_pretraining/detr_finetuned_100umc/checkpoint0300.pth',
                     'model_mode': 'detr_umc', # detr_rm, detr_umc, detr_dl, mdetr
                     }

    args = Namespace(**args_dict)
    eval_checkpoint(args, giou_thresh=0.5, dist_thresh=1.0)
