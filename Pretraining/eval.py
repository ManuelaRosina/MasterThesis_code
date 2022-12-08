from torchvision.models import vgg16, resnet18, resnet34, resnet50, resnet101
import torch
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
import utils
from data import DeepLesion_eval, UnmatchedLesionData, LesionData
from torch.utils.data import DataLoader
import time
import datetime

def center_distcance(pred, tgt):
    if len(tgt) > 2:
        tgt = tgt[:2,:]
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
    rel = torch.log(dist/(lesion_length*0.5) + 1)
    values, idx = dist.min(axis=1)
    
    return rel, values, idx
    

def useable(pred, tgt):
    """
    if len(tgt) > 2:
        tgt = tgt[:2,:]
    tgt_center = tgt[:, 2:] - tgt[:, :2]
    
    
    upper = torch.all(pred[:, :2].le(tgt_center) , dim=1)
    lower = torch.all(pred[:, 2:].gt(tgt_center) , dim=1)
    
    inside = torch.all(torch.stack((upper,lower)), dim=0)
    
    
    if torch.any(inside[:len(tgt)]):
        print(tgt.shape)
        print("tgt_center", tgt_center)
        print("pred", pred[:, :2])
        print("pred", pred[:, 2:])
        print("upper", upper)
        print("lower", lower)
        print(inside)
        print(inside[:len(tgt)])
    """
    pred_center = pred[:, 2:] - pred[:, :2]
    
    upper = torch.all(tgt[:2, :2].le(pred_center) , dim=1)
    lower = torch.all(tgt[:2, 2:].gt(pred_center) , dim=1)
    
    inside = torch.all(torch.stack((upper,lower)), dim=0)
    
    if torch.any(inside):
        print(tgt.shape)
        print("pred_center", pred_center)
        print("pred", pred[:, :2])
        print("pred", pred[:, 2:])
        print("upper", upper)
        print("lower", lower)
        print(inside)

    

# return number of tp, fp and fn based on giou
# if giou > 0.5 with ground truth count as hit
def evaluate(pred, tgt, giou_thresh=0.5, dist_thresh=0.3):
    # calculate giou for pred with match ground truth
    giou = utils.generalized_box_iou(pred, tgt) #The boxes should be in [x0, y0, x1, y1] format
    #Returns a[N, M] pairwise matrix, where N = len(pred) and M = len(tgt)
    #print(giou)
    values, idx = giou.max(axis=1)
    #print(idx.unique())
    hits = (values >= giou_thresh)[:len(tgt)]
    #print(hits)
    tp = int(torch.sum((hits == True)))
    fp = len(hits) - len(idx.unique()) # idx are the indexes of ground truth boxes matched with prediction.
    # If one (or more) is missing a ground truth box was not predicted
    fn = int(len(tgt) - tp)
    
    #usable = useable(pred, tgt)
    
    dist = center_distcance(pred, tgt)
    d_values, idx = dist.min(axis=1)
    hits = (d_values >= dist_thresh)
    d_tp = int(torch.sum((hits == True)))
    d_fp = len(tgt) - len(idx.unique()) #TODO: is this really right?
    d_fn = d_fp #int(len(tgt) - tp)
    
    if len(tgt) >= 3:
        print("three target boxes")

    return tp, fp, fn, values[:len(tgt)], dist, d_tp, d_fp, d_fn
    
    
def evaluate2(pred, tgt, giou_thresh=0.5):
    correct_pred, total_pred = 0, 0
    tp, fp, fn = 0, 0, 0

    # calculate giou for pred with match ground truth
    giou = utils.generalized_box_iou(pred, tgt) #The boxes should be in [x0, y0, x1, y1] format
    #Returns a[N, M] pairwise matrix, where N = len(pred) and M = len(tgt)
    values, idx = giou.max(axis=1)
    hits = (values >= giou_thresh)
    #print(giou.shape)
    #print(values.shape)
    
    # accuracy
    if int(torch.sum((hits == True))) > 0:
        correct_pred = int(torch.sum((hits == True)))
    total_pred = len(tgt[:2,:])
    
    if len(tgt) > 1:
        # precision
        tp = int(torch.sum((hits == True)))
        fn = len(tgt) - len(idx.unique()) # idx are the indexes of ground truth boxes matched with prediction.
    # If one (or more) is missing a ground truth box was not predicted
        fp = fn # if there are two pred matched to the same tgt there is automatically a fp
        
    return tp, fp, fn, correct_pred, total_pred, values[:len(tgt)]
    
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
    total_pred = len(tgt[:2,:])
    
    if len(tgt) > 1:
        # precision
        tp = int(torch.sum((hits == True)))
        fn = len(tgt) - len(idx.unique()) # idx are the indexes of ground truth boxes matched with prediction.
    # If one (or more) is missing a ground truth box was not predicted
        fp = fn # if there are two pred matched to the same tgt there is automatically a fp
        
    return tp, fp, fn, correct_pred, total_pred, dist[:len(tgt)]
        
def dist_iou(pred, tgt, thresh=0.5):
    correct_pred, total_pred = 0, 0
    tp, fp, fn = 0, 0, 0

    # calculate giou for pred with match ground truth
    giou = utils.dist_iou(pred, tgt) #The boxes should be in [x0, y0, x1, y1] format
    #Returns a[N, M] pairwise matrix, where N = len(pred) and M = len(tgt)
    values, idx = giou.max(axis=1)
    hits = (values >= thresh)
    
    # accuracy
    if int(torch.sum((hits == True))) > 0:
        correct_pred = int(torch.sum((hits == True)))
    total_pred = len(tgt[:2,:])
    
    if len(tgt) > 1:
        # precision
        tp = int(torch.sum((hits == True)))
        fn = len(tgt) - len(idx.unique()) # idx are the indexes of ground truth boxes matched with prediction.
    # If one (or more) is missing a ground truth box was not predicted
        fp = fn # if there are two pred matched to the same tgt there is automatically a fp
        
    return tp, fp, fn, correct_pred, total_pred, giou

def precision_recall(tp, fp, fn):
    # precision and recall computation
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall
    

class Backbone(nn.Module):
    def __init__(self, num_channels=512, net_type='vgg'):
        super().__init__()
        self.num_channels = num_channels
        if net_type == 'vgg':
            self.backbone = vgg16()
            del self.backbone.classifier

        if 'resnet' in net_type:
            if net_type == 'resnet18':
                self.backbone = resnet18()
            elif net_type == 'resnet34':
                self.backbone = resnet34()
            elif net_type == 'resnet50':
                self.backbone = resnet50()
            elif net_type == 'resnet101':
                self.backbone = resnet101()
            else:
                print("ERROR")
            del self.backbone.fc

        # create conversion layer
        if net_type == 'resnet50' or net_type == 'resnet101':
            self.conv = nn.Conv2d(2048, self.num_channels, 1)
        else:
            self.conv = nn.Conv2d(512, self.num_channels, 1)

        return_layers = {"avgpool": "0"}
        self.body = IntermediateLayerGetter(self.backbone, return_layers=return_layers)

        self.flat = nn.Flatten()
        if net_type == 'vgg':
            self.linear1 = nn.Linear(25088, 128)
        if 'resnet' in net_type:
            self.linear1 = nn.Linear(self.num_channels, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, 8)


    def forward(self, inputs):
        xs = self.body(inputs)
        out = []
        for name, x in xs.items():
            x = self.conv(x)
            x = self.flat(x)
            x = self.linear1(x).relu()
            x = self.linear2(x).relu()
            x = self.linear3(x).relu()
            x = self.linear4(x).sigmoid()

            out.append(x.reshape((-1, 2, 4)))

        return out[-1]

def checkpoint_evaluation(net_type, checkpoint_path, model_mode, giou_thresh=0.5, dist_thresh=1.0):
    path_chansey = '/output/'
    device = torch.device("cuda")
    backbone = Backbone(net_type=net_type)
    #print(backbone)
    backbone_checkpoint = path_chansey+checkpoint_path
    checkpoint = torch.load(backbone_checkpoint, map_location=torch.device('cpu'))
    backbone.load_state_dict(checkpoint["model_state_dict"])
    backbone.to(device)
    backbone.eval()

    print("evaluating checkpoint ", backbone_checkpoint)
    
    if model_mode == 'rm':
        print("report matched dataset")
        
        test_dataset = LesionData(path_chansey + 'Data/FromReports/', path_chansey, split='test')
    elif model_mode == 'dl':
        print("DeepLesion dataset")
        test_dataset = DeepLesion_eval(path_chansey + 'Pretraining/', path_chansey + 'Data/volumes/', split='test')
    elif model_mode == 'umc':
        print("umc unmatched dataset")
        test_dataset = UnmatchedLesionData(path_chansey + 'Data/UnmatchedGSPS/', path_chansey, split='test')

    #test_dataset = DeepLesion_eval(path_chansey + 'Pretraining/', path_chansey + 'Data/volumes/', split='test')
    #test_dataset = UnmatchedLesionData(path_chansey + 'Data/UnmatchedGSPS/', path_chansey, split='test')
    val_sampler = torch.utils.data.RandomSampler(test_dataset)
    batch_sampler_val = torch.utils.data.BatchSampler(val_sampler, batch_size=1, drop_last=False)
    eval_data_loader = DataLoader(test_dataset, batch_sampler=batch_sampler_val, collate_fn=utils.collate_fn)

    total_tp, total_fp, total_fn = 0, 0, 0
    total_d_tp, total_d_fp, total_d_fn = 0, 0, 0
    idxs, idxs2 = [], []
    gious = []
    dists = []
    total_correct_pred, total_total_pred = 0, 0
    d_total_tp, d_total_fp, d_total_fn = 0, 0, 0
    d_total_correct_pred, d_total_total_pred = 0, 0

    print("start evaluation")
    start_time = time.time()
    print("using giou thresh: ", giou_thresh, " and dist thresh: ", dist_thresh)

    for i, batch_dict in enumerate(eval_data_loader):
        if i % 100 == 0:
            print(i)
        samples = batch_dict["samples"].to(device)
        targets = batch_dict["targets"]
        targets = utils.targets_to(targets, device)
        tgt = [t['boxes'] for t in targets][0]
        orig_size = [t['orig_size'] for t in targets][0]
        idx = [t['image_id'] for t in targets][0]

        prediction = backbone(samples)
        prediction = utils.rescale_bboxes(prediction, orig_size).squeeze()

        """
        tp, fp, fn, giou_values, dist, d_tp, d_fp, d_fn = evaluate(prediction, tgt, giou_thresh=giou_thresh)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_d_tp += d_tp
        total_d_fp += d_fp
        total_d_fn += d_fn
        #print(giou_values.tolist())
        gious.extend(giou_values.tolist())
        dists.extend(dist.tolist())
        """
        tp, fp, fn, correct_pred, total_pred, giou = evaluate2(prediction, tgt, giou_thresh=giou_thresh)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_correct_pred += correct_pred
        total_total_pred += total_pred
        gious.extend(giou.tolist())
        
        #print("dist_eval: ", dist_evaluate(prediction, tgt, dist_thresh=0.3))
        
        tp, fp, fn, correct_pred, total_pred, dist = dist_evaluate(prediction, tgt, dist_thresh=dist_thresh)
        d_total_tp += tp
        d_total_fp += fp
        d_total_fn += fn
        d_total_correct_pred += correct_pred
        d_total_total_pred += total_pred
        dists.extend(dist.tolist())
        
        if fn > 0:
            idxs.append(idx)
        else:
            idxs2.append(idx)

    #precision, recall = precision_recall(total_tp, total_fp, total_fn)
    #print('precision: ', precision)
    #print('recall: ', recall)
    print('correct_pred: ', total_correct_pred, ' total_pred: ', total_total_pred)
    print('accuracy: ', total_correct_pred/total_total_pred)
    print("distance based:")
    print('correct_pred: ', d_total_correct_pred, ' total_pred: ', d_total_total_pred)
    print('accuracy: ', d_total_correct_pred/d_total_total_pred)
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
    
checkpoints = [
#('resnet18', 'Final_Backbone_Pretraining/Resnet_18/checkpoint0050.pth'),
#('resnet18', 'Final_Backbone_Pretraining/Resnet_18_whithout_imagenet/checkpoint0050.pth'),
#('resnet34', 'Final_Backbone_Pretraining/Resnet_34/checkpoint0050.pth'),
#('resnet34', 'Final_Backbone_Pretraining/Resnet_34_whithout_imagenet/checkpoint0050.pth'),
#('resnet50', 'Final_Backbone_Pretraining/Resnet_50/checkpoint0050.pth'),
#('resnet50', 'Final_Backbone_Pretraining/Resnet_50_whithout_imagenet/checkpoint0050.pth'),
('resnet101', 'Final_Backbone_Pretraining/Resnet_101/checkpoint0050.pth'),
#('resnet101', 'Final_Backbone_Pretraining/Resnet_101_whithout_imagenet/checkpoint0050.pth'),
#('vgg', 'Final_Backbone_Pretraining/Vgg_16/checkpoint0050.pth'),
#('vgg', 'Final_Backbone_Pretraining/Vgg_16_without_pretraining/checkpoint0050.pth'),
#('resnet50', 'Final_Backbone_Pretraining/resnet50_2/checkpoint0019.pth'),
#('resnet50', 'Final_Backbone_Pretraining/resnet50_2/checkpoint0021.pth'),
#('resnet50', 'Final_Backbone_Pretraining/resnet50_2/checkpoint0029.pth'),
#('resnet50', 'Final_Backbone_Pretraining/resnet50_2/checkpoint0031.pth'),
#('resnet50', 'Final_Backbone_Pretraining/resnet50_3/checkpoint0050.pth'),
#('resnet50', 'Final_Backbone_Pretraining/resnet50_3/checkpoint0011.pth'),
#('resnet50', 'Final_Backbone_Pretraining/resnet50_3/checkpoint0005.pth'),
#('resnet50', 'Final_Backbone_Pretraining/resnet50_4/checkpoint0050.pth'),
#('resnet50', 'Final_Backbone_Pretraining/resnet50_4/checkpoint0011.pth'),
#('resnet50', 'Final_Backbone_Pretraining/resnet50_4/checkpoint0005.pth'),
#('resnet50', 'Final_Backbone_Pretraining/resnet50_5/checkpoint0050.pth'),
#('resnet50', 'Final_Backbone_Pretraining/resnet50_5/checkpoint0025.pth'),
#('resnet50', 'Final_Backbone_Pretraining/resnet50_giou/checkpoint0050.pth'),
#('resnet50', 'Final_Backbone_Pretraining/resnet50_giou/checkpoint0025.pth'),
#('resnet50', 'Final_Backbone_Pretraining/resnet50_l1/checkpoint0050.pth'),
#('resnet50', 'Final_Backbone_Pretraining/resnet50_l1/checkpoint0025.pth')
#('resnet101', 'Final_Backbone_Pretraining/resnet101_dl_20%umc/checkpoint0050.pth'),
#('resnet101', 'Final_Backbone_Pretraining/resnet101_dl_20%umc_finetune/checkpoint0020.pth')
('resnet101', 'Final_Backbone_Pretraining/resnet101_umc_only/checkpoint0050.pth'),
#('resnet101', 'Final_Backbone_Pretraining/resnet101_umc_only/checkpoint0045.pth'),
#('resnet101', 'Final_Backbone_Pretraining/resnet101_umc_only/checkpoint0031.pth')
]

for net_type, checkpoint_path in checkpoints:
    checkpoint_evaluation(net_type, checkpoint_path, 'rm', giou_thresh=0.5, dist_thresh=1.0)
    
for net_type, checkpoint_path in checkpoints:
    checkpoint_evaluation(net_type, checkpoint_path, 'dl', giou_thresh=0.5, dist_thresh=1.0)

"""
path_chansey = '/output/'
test_dataset = DeepLesion_eval(path_chansey + 'Pretraining/', path_chansey + 'Data/volumes/', split='test')
val_sampler = torch.utils.data.RandomSampler(test_dataset)
batch_sampler_val = torch.utils.data.BatchSampler(val_sampler, batch_size=1, drop_last=False)
eval_data_loader = DataLoader(test_dataset, batch_sampler=batch_sampler_val, collate_fn=utils.collate_fn)

len_1, len_2, len_3 = 0, 0, 0

for i, batch_dict in enumerate(eval_data_loader):
    targets = batch_dict["targets"]
    #targets = utils.targets_to(targets, device)
    tgt = [t['boxes'] for t in targets][0]
    
    if len(tgt) == 1:
        len_1 += 1
    elif len(tgt) == 2:
        len_2 += 1
    elif len(tgt) == 3:
        len_3 += 1
        
print("DeepLesion test")        
print("len_1: ", len_1, " len_2: ", len_2, " len_3: ", len_3)


test_dataset = DeepLesion_eval(path_chansey + 'Pretraining/', path_chansey + 'Data/volumes/', split='val')
val_sampler = torch.utils.data.RandomSampler(test_dataset)
batch_sampler_val = torch.utils.data.BatchSampler(val_sampler, batch_size=1, drop_last=False)
eval_data_loader = DataLoader(test_dataset, batch_sampler=batch_sampler_val, collate_fn=utils.collate_fn)

len_1, len_2, len_3 = 0, 0, 0

for i, batch_dict in enumerate(eval_data_loader):
    targets = batch_dict["targets"]
    #targets = utils.targets_to(targets, device)
    tgt = [t['boxes'] for t in targets][0]
    
    if len(tgt) == 1:
        len_1 += 1
    elif len(tgt) == 2:
        len_2 += 1
    elif len(tgt) == 3:
        len_3 += 1
        
print("DeepLesion val")        
print("len_1: ", len_1, " len_2: ", len_2, " len_3: ", len_3)


test_dataset = DeepLesion_eval(path_chansey + 'Pretraining/', path_chansey + 'Data/volumes/', split='train')
val_sampler = torch.utils.data.RandomSampler(test_dataset)
batch_sampler_val = torch.utils.data.BatchSampler(val_sampler, batch_size=1, drop_last=False)
eval_data_loader = DataLoader(test_dataset, batch_sampler=batch_sampler_val, collate_fn=utils.collate_fn)

len_1, len_2, len_3 = 0, 0, 0

for i, batch_dict in enumerate(eval_data_loader):
    targets = batch_dict["targets"]
    #targets = utils.targets_to(targets, device)
    tgt = [t['boxes'] for t in targets][0]
    
    if len(tgt) == 1:
        len_1 += 1
    elif len(tgt) == 2:
        len_2 += 1
    elif len(tgt) == 3:
        len_3 += 1
        
print("DeepLesion train")        
print("len_1: ", len_1, " len_2: ", len_2, " len_3: ", len_3)
"""
