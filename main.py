import sys

import torch
from torch import tensor

from argparse import Namespace
from model import build_model
from utils import build_postprocessor
import prepare_dataset
from prepare_dataset import LesionData, load_test_image, CocoDetection, DeepLesion, UnmatchedLesionData
from transformers import RobertaModel, RobertaTokenizerFast, BertTokenizerFast
from torch.utils.data import DataLoader,  ConcatDataset
import utils
from functools import partial
import time
import datetime
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Iterable
from evaluation import Evaluator
import json
from metrics import MetricLogger, SmoothedValue, adjust_learning_rate
import math
import os
from os import listdir
from os.path import isfile, join


def save_on_master(*args, **kwargs):
    """Utility function to save only from the main process"""
    torch.save(*args, **kwargs)

def pos_map(targets, num_queries):
    pos_map = []
    for target in targets:
        for i in range(num_queries):
            if i < len(target['positive map']):
                pos_map.append(target['positive map'][i])
            else:
                pos_map.append(torch.zeros(256))
    return torch.stack(pos_map)


def train_one_epoch(
    model: torch.nn.Module,
    criterion: Optional[torch.nn.Module],
    weight_dict: Dict[str, float],
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    lr_scheduler,
    max_norm: float = 0,
):
    model.train()
    if criterion is not None:
        criterion.train()
    metric_logger = MetricLogger(delimiter="  ", path=args.logging_path)
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("lr_backbone", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("lr_text_encoder", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 100

    num_training_steps = int(len(data_loader) * args.epochs)
    for i, batch_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        curr_step = epoch * len(data_loader) + i

        #image, targets = dataset[5]
        #captions = targets['sentence']

        samples = batch_dict["samples"].to(device)
        targets = batch_dict["targets"]
        targets = utils.targets_to(targets, device)
        if args.dataset == 'detr_rm':
            captions = None
        else:
            captions = [t['sentence'] for t in targets] if 'sentence' in targets[0] else None
        positive_map = batch_dict["positive map"].to(device) if "positive map" in batch_dict else None
        #captions = None
        #positive_map = None

        outputs, memory_cache = model(samples, captions)
        #outputs, memory_cache = model(samples, None)
        #print(outputs["pred_boxes"])

        loss_dict = {}
        if criterion is not None:
            loss_dict.update(criterion(outputs, targets, positive_map))
            #loss_dict.update(criterion(outputs, targets, positive_map=None))

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = loss_dict
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        #lr_scheduler.step()

        adjust_learning_rate(
            optimizer,
            epoch,
            curr_step,
            num_training_steps=num_training_steps,
            args=args,
        )

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_backbone=optimizer.param_groups[1]["lr"])
        metric_logger.update(lr_text_encoder=optimizer.param_groups[2]["lr"])
    # gather the stats from all processes
    print("Averaged stats:", metric_logger)
    metric_logger.updateTensorboard(epoch)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: Optional[torch.nn.Module],
    postprocessor,
    weight_dict: Dict[str, float],
    data_loader,
    evaluator,
    device: torch.device,
    args,
    epoch,
):
    model.eval()
    if criterion is not None:
        criterion.eval()

    metric_logger = MetricLogger(delimiter="  ", path=args.logging_path)
    header = "Test:"

    for batch_dict in metric_logger.log_every(data_loader, 10, header):
        samples = batch_dict["samples"].to(device)
        targets = batch_dict["targets"]
        targets = utils.targets_to(targets, device)
        if args.dataset == 'detr_rm':
            captions = None
        else:
            captions = [t['sentence'] for t in targets] if 'sentence' in targets[0] else None
        positive_map = batch_dict["positive map"].to(device) if "positive map" in batch_dict else None
        #captions = None
        #positive_map = None

        outputs, memory_cache = model(samples, captions)
        #outputs, memory_cache = model(samples, None)

        #print(outputs["pred_boxes"])
        #orig_target_sizes = torch.tensor([list(targets[0]["orig_size"])])
        #single_output = {'pred_logits': outputs['pred_logits'][0,:,:].unsqueeze(0), 'pred_boxes':outputs['pred_boxes'][0,:,:].unsqueeze(0)}
        #single_cache = {'text': memory_cache['text'][0], 'tokenized':memory_cache['tokenized']}

        #results, final_results = postprocessor(single_output, single_cache, orig_target_sizes)
        #print(results)
        #print(final_results)
        #utils.print_image_prediction(samples.tensors[0,:,:,:].permute(1, 2, 0), final_results[0])

        loss_dict = {}
        if criterion is not None:
            #loss_dict.update(criterion(outputs, targets, positive_map=None))
            loss_dict.update(criterion(outputs, targets, positive_map))

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = loss_dict
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )
        orig_target_sizes = torch.tensor([list(t["orig_size"]) for t in targets])
        results, final_results = postprocessor(outputs, memory_cache, orig_target_sizes)
        #results = postprocessor(outputs, orig_target_sizes)
        #res = {target["image_id"]: output for target, output in zip(targets, results)}
        evaluator.update(results, targets)

    # gather the stats from all processes
    print("Averaged stats:", metric_logger)
    metric_logger.updateTensorboard(epoch, mode='test')

    score = evaluator.summarize()

    # accumulate predictions from all images

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats['eval_score'] = score
    return stats


def save_checkpoint(output_dir, epoch, model, optimizer, save_freq=2):
    if (epoch + 1) % save_freq == 0:
        checkpoint_paths = output_dir + f"checkpoint{epoch:04}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_paths)
        

def main(args):
    device = torch.device(args.device)
    postprocessor = build_postprocessor()
    
    # check if arguments are right for mdetr mode
    if args.dataset == 'reportmatched':
        if args.ce_loss_coef == 0:
            args.ce_loss_coef = 1
    else:
    # make sure arguments are right for detr mode
        args.freeze_text_encoder = True
        args.ce_loss_coef = 0
    

    output_dir = Path(args.output_dir)
    #x = NestedTensor.from_tensor_list([torch.randn(3, 224, 224)])
    #print(x.tensors.dtype)
    model, criterion, weight_dict = build_model(args)
    model.to(device)
    #print([n for n, p in model.named_parameters()])

    #tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    #tokenizer = BertTokenizerFast.from_pretrained("giacomomiolo/biobert_reupload") #AutoTokenizer.from_pretrained("giacomomiolo/biobert_reupload")
    
    if args.text_encoder_type == 'roberta-base':
            tokenizer = RobertaTokenizerFast.from_pretrained(args.path_chansey + "Data/Pretrain_files/" + args.text_encoder_type, local_files_only=True)
    elif args.text_encoder_type == 'biobert':
        tokenizer = BertTokenizerFast.from_pretrained('giacomomiolo/biobert_reupload')
    elif args.text_encoder_type == 'robbert':
        tokenizer = RobertaTokenizerFast.from_pretrained('pdelobelle/robbert-v2-dutch-base')
    elif 'bertje' in args.text_encoder_type:
        tokenizer = BertTokenizerFast.from_pretrained('GroNLP/bert-base-dutch-cased')
            
    #train_dataset = LesionData(args.path_chansey + 'Data/FromReports/', args.path_chansey, tokenizer, split='train')
    #val_dataset = LesionData(args.path_chansey + 'Data/FromReports/', args.path_chansey, tokenizer, split='val') #LesionData(args.path_chansey + 'Data/FromReports/', tokenizer, args.path_chansey + args.img_dir) #CocoDetection('/output/Data/train2014/', '/output/Data/finetune_refcoco_testA.json', tokenizer)
    #sampler = torch.utils.data.RandomSampler(dataset)
    #train_sampler = torch.utils.data.RandomSampler(train_dataset)
    #val_sampler = torch.utils.data.RandomSampler(val_dataset)
    #batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=False)
    #batch_sampler_eval = torch.utils.data.BatchSampler(val_sampler, args.batch_size, drop_last=False)
    #train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, collate_fn=partial(utils.collate_fn, False))
    #eval_data_loader = DataLoader(val_dataset, batch_sampler=batch_sampler_eval, collate_fn=partial(utils.collate_fn, False))
    
    if args.dataset == 'dl':
        train_dataset = DeepLesion(args.path_chansey + 'Pretraining/', args.path_chansey + 'Data/volumes/')
        val_dataset = DeepLesion(args.path_chansey + 'Pretraining/', args.path_chansey + 'Data/volumes/', split='val')
    elif args.dataset == 'umcmixed':
        dl_train_dataset = DeepLesion(args.path_chansey + 'Pretraining/', args.path_chansey + 'Data/volumes/')
        dl_val_dataset = DeepLesion(args.path_chansey + 'Pretraining/', args.path_chansey + 'Data/volumes/', split='val')
        umc_train_dataset = UnmatchedLesionData(args.path_chansey + 'Data/UnmatchedGSPS/', args.path_chansey, portion=args.umc_data_portion)
        umc_val_dataset = UnmatchedLesionData(args.path_chansey + 'Data/UnmatchedGSPS/', args.path_chansey, split='val')
    
        train_dataset = ConcatDataset([dl_train_dataset, umc_train_dataset])
        val_dataset = ConcatDataset([dl_val_dataset, umc_val_dataset])
    elif args.dataset == 'umc':
        train_dataset = UnmatchedLesionData(args.path_chansey + 'Data/UnmatchedGSPS/', args.path_chansey, portion=args.umc_data_portion)
        val_dataset = UnmatchedLesionData(args.path_chansey + 'Data/UnmatchedGSPS/', args.path_chansey, split='val')
    elif args.dataset == 'reportmatched' or args.dataset == 'detr_rm':
        train_dataset = LesionData(args.path_chansey + 'Data/FromReports/', args.path_chansey, tokenizer, split='train')
        val_dataset = LesionData(args.path_chansey + 'Data/FromReports/', args.path_chansey, tokenizer, split='val')
        
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    val_sampler = torch.utils.data.RandomSampler(val_dataset)
    batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=False)
    batch_sampler_val = torch.utils.data.BatchSampler(val_sampler, args.batch_size, drop_last=False)
    
    if args.dataset == 'dl' or args.dataset == 'umc' or args.dataset == 'umcmixed':
        train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn_deep_lesion)
        eval_data_loader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, collate_fn=utils.collate_fn_deep_lesion)
    else:
        train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, collate_fn=partial(utils.collate_fn, False))
        eval_data_loader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, collate_fn=partial(utils.collate_fn, False))
    
    
    param_dicts = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if ("backbone" not in n or "resizer" in n) and "text_encoder" not in n and p.requires_grad
                ]
            },
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and "text_encoder" not in n and "resizer" not in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
            {
                "params": [p for n, p in model.named_parameters() if "text_encoder" in n and p.requires_grad],
                "lr": args.text_encoder_lr,
            },
        ]

    #for d in param_dicts:
    #    print(d.keys())
    #print('backbone:')
    #print([n for n, p in model.named_parameters() if "backbone" in n and p.requires_grad])
    #print('text encoder: ')
    #print([n for n, p in model.named_parameters() if "text_encoder" in n and p.requires_grad])

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer in ["adam", "adamw"]:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
        
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
        
    if args.start_epoch != 0:
        print('loading previous state dict...')
        last_epoch = args.start_epoch -1
        checkpoint = torch.load(args.path_chansey + args.output_dir + f"checkpoint{last_epoch:04}.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print('loaded model and optimizer')
        
    if not os.path.exists(args.path_chansey + args.output_dir):
        # create the directory to store the checkpoints
        if args.output_dir:
            os.makedirs(args.path_chansey + args.output_dir)
            print("created dir ", args.path_chansey + args.output_dir)
    else:
        # check if there are already checkpoints and load the last one
        checkpoint_files = [f for f in listdir(args.path_chansey + args.output_dir) if isfile(join(args.path_chansey + args.output_dir, f))]
        if checkpoint_files:
            last_checkpoint = sorted(checkpoint_files)[-1]
            
            checkpoint = torch.load(args.path_chansey + args.output_dir + last_checkpoint)
            model.load_state_dict(checkpoint["model_state_dict"])
            #print(checkpoint["optimizer_state_dict"].keys())
            #print(optimizer.state_dict().keys())
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            args.start_epoch = checkpoint["epoch"]+1
            print("loaded checkpoint ", last_checkpoint, " from ", args.path_chansey + args.output_dir)

    #train(model, train_dataloader, optimizer)

    #for i in range(4):
    # image, targets = dataset[0]
    # plot_test_image(model, dataset, postprocessor, image, targets)
    #utils.print_image_prediction(image.permute(1, 2, 0), targets)

    #image, targets = load_test_image(tokenizer)
    #utils.print_image_prediction_2(image.permute(1, 2, 0), targets)
    #plot_test_image(model, dataset, postprocessor, image, targets)

    print("Start training")
    start_time = time.time()
    best_metric = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        print(f"Starting epoch {epoch}")
        train_stats = train_one_epoch(model, criterion, weight_dict, train_dataloader, optimizer, device, epoch, args, lr_scheduler)
        #lr_scheduler.step()

        if epoch % args.eval_skip == 0:
            test_stats = {}
            print("Evaluating...")
            evaluator = Evaluator()
            curr_test_stats = evaluate(model, criterion, postprocessor, weight_dict, eval_data_loader, evaluator, device, args, epoch)
            test_stats.update({k: v for k, v in curr_test_stats.items()})
            #plot_test_image(model, dataset, postprocessor)
        else:
            test_stats = {}

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir:
            save_checkpoint(args.path_chansey + args.output_dir, epoch, model, optimizer, save_freq=args.save_freq)

        #print(test_stats)

    """
        if epoch % args.eval_skip == 0:
            #metric = np.mean([v[1] for k, v in test_stats.items() if "coco_eval_bbox" in k])
    
            if args.output_dir and metric > best_metric:
                best_metric = metric
                checkpoint_paths = [output_dir / "BEST_checkpoint.pth"]
                # extra checkpoint before LR drop and every 100 epochs
                for checkpoint_path in checkpoint_paths:
                    save_on_master(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "args": args,
                        },
                        checkpoint_path,
                    )
    """
    if args.output_dir:
        #save_checkpoint(args.path_chansey + args.output_dir, epoch, model, optimizer)
        output_dir = args.path_chansey + args.output_dir
        checkpoint_paths = output_dir + f"checkpoint{epoch:04}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_paths)
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    #for i, para in enumerate(model.parameters()):
    #    print(f'{i+1}th parameter tensor:', para.shape)
    #    print(para)
    #    print(para.grad)
    #    print()


def plot_test_image(model, dataset, postprocessor):
    model.eval()
    image, targets = dataset[5]
    captions = targets['sentence']

    outputs, memory_cache = model([image], [captions])
    #print(outputs['pred_boxes'])
    #print(memory_cache.keys())
    orig_target_sizes = torch.tensor([list(targets["orig_size"])])
    results, final_results = postprocessor(outputs, memory_cache, orig_target_sizes)
    #results = postprocessor(outputs, orig_target_sizes)
    #print(results)
    print(final_results)
    utils.print_image_prediction(image.permute(1, 2, 0), final_results[0])


def test(args):
    #tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    #image, targets = load_test_image(tokenizer)
    #utils.print_image_prediction_2(image.permute(1, 2, 0), targets)
    model, criterion, weight_dict = build_model(args)
    #print(model.transformer.decoder.norm)
    #for n, p in model.named_parameters():
    #    if ("backbone" not in n or "resizer" in n) and "text_encoder" not in n and p.requires_grad:
    #        print(n)

    inputs = torch.randn(1, 3, 224, 224)
    outputs,cache = model(inputs, ["this is a test"])
    #print(outputs)
    targets = {}
    targets['boxes'] = torch.Tensor([(0.5, 0.5, 0.1, 0.1)])
    targets['scaled boxes'] = torch.Tensor([(0.5, 0.5, 0.1, 0.1)])
    targets["tokens_positive"] = [[(10, 14)]]
    targets['labels'] = torch.Tensor([1])

    positive_map = prepare_dataset.create_positive_map(cache['tokenized'], targets["tokens_positive"])
    # print(positive_map)
    loss_dict = {}
    loss_dict.update(criterion(outputs, [targets], positive_map))
    print(loss_dict)

def test2(args):
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    #data = CocoDetection('Data/train2014/', 'Data/finetune_refcoco_testA.json', tokenizer)
    dataset = LesionData(args.path_chansey + 'Data/FromReports/', tokenizer, args.path_chansey + args.img_dir)
    for i, (img, targets) in enumerate(dataset):
        boxes = utils.box_cxcywh_to_xyxy(targets['scaled boxes'])
        if not (boxes[:, 2:] >= boxes[:, :2]).all():
            print(i, targets['sentence_id'], boxes)
            print(targets['boxes'], targets['orig_size'])
            print()

def compute_stats(args):
    #path_chansey = '/output/'
    #tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    #dataset = DeepLesion(path_chansey + 'Pretraining/', path_chansey + 'Data/volumes/')
    dataset = UnmatchedLesionData(args.path_chansey + 'Data/UnmatchedGSPS/', args.path_chansey, split='all', portion=1)
    loader = DataLoader(dataset,
                         batch_size=10,
                         num_workers=0,
                         shuffle=False, collate_fn=utils.collate_fn_deep_lesion)

    mean = 0.
    std = 0.
    for batch_dict in loader:
        #print(batch_dict)
        images = batch_dict["samples"]
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        images = images.float()
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    
    print("mean: ", mean)
    print("std: ", std)   
    
def test_dataset(args):
    train_dataset = UnmatchedLesionData(args.path_chansey + 'Data/UnmatchedGSPS/', args.path_chansey, portion=1)
    val_dataset = UnmatchedLesionData(args.path_chansey + 'Data/UnmatchedGSPS/', args.path_chansey, split='val')
    test_dataset = UnmatchedLesionData(args.path_chansey + 'Data/UnmatchedGSPS/', args.path_chansey, split='test')
    
    #train_dataset = ConcatDataset([dl_train_dataset, umc_train_dataset])
    #val_dataset = ConcatDataset([dl_val_dataset, umc_val_dataset])
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    val_sampler = torch.utils.data.RandomSampler(val_dataset)
    test_sampler = torch.utils.data.RandomSampler(test_dataset)
    batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=False)
    batch_sampler_val = torch.utils.data.BatchSampler(val_sampler, args.batch_size, drop_last=False)
    batch_sampler_test = torch.utils.data.BatchSampler(test_sampler, args.batch_size, drop_last=False)
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn_deep_lesion)
    eval_data_loader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, collate_fn=utils.collate_fn_deep_lesion)   
    test_data_loader = DataLoader(test_dataset, batch_sampler=batch_sampler_test, collate_fn=utils.collate_fn_deep_lesion)             
    
    print("train set:")
    for i, batch_dict in enumerate(train_dataloader): 
        if i%100 == 0:
            print(i)
        samples = batch_dict["samples"]
        targets = batch_dict["targets"]
        
    print()
    print("val set:")
    for i, batch_dict in enumerate(eval_data_loader): 
        if i%100 == 0:
            print(i)
        samples = batch_dict["samples"]
        targets = batch_dict["targets"]
        
    print()    
    print("tests set:")
    for i, batch_dict in enumerate(test_data_loader): 
        if i%100 == 0:
            print(i)
        samples = batch_dict["samples"]
        targets = batch_dict["targets"]
        
def dataset_len(args):
    dataset = UnmatchedLesionData(args.path_chansey + 'Data/UnmatchedGSPS/', args.path_chansey, split='all')
    print("Unmatched: ", len(dataset))
    dataset = LesionData(args.path_chansey + 'Data/FromReports/', args.path_chansey, None, split='all')
    print("ReportMatched: ", len(dataset))

if __name__ == "__main__":
    args_dict = {'lr_backbone': 1e-5,
                 'text_encoder_lr': 1e-5,
                 'lr': 1e-4,
                 'min_lr': 1e-6,
                 'lr_drop': 200,
                 'weight_decay': 1e-4,
                 'hidden_dim': 256,
                 'num_channels': 512,
                 'num_classes': 255,
                 'num_queries': 3,
                 'text_encoder_type': 'bertje', #"roberta-base", #'giacomomiolo/biobert_reupload',
                 'backbone_type': 'resnet101',
                 'img_back_pretrained': True,
                 'freeze_text_encoder': False,  # True,
                 'freeze_image_encoder': False,
                 'dropout': 0.1,
                 'nheads': 8,
                 'dim_feedforward': 2048,
                 'enc_layers': 2,
                 'dec_layers': 2,
                 'pre_norm': False,
                 'pass_pos_and_query': True,
                 'aux_loss': False,
                 'contrastive_loss_hdim': 64,
                 'contrastive_align_loss': False,
                 'contrastive_align_loss_coef': 1,
                 'temperature_NCE': 0.07,
                 'set_cost_class': 1,
                 'set_cost_bbox': 5,
                 'set_cost_giou': 2,
                 'ce_loss_coef': 0,
                 'bbox_loss_coef': 5,
                 'giou_loss_coef': 2,
                 'eos_coef': 0.1,  # help="Relative classification weight of the no-object class"
                 'batch_size': 4,
                 'optimizer': 'adam',  # 'sgd',
                 'output_dir': 'Final_Detr_pretraining/Detr_UMC_20_mixed_new/', #"Test_13/",  # help="path where to save, empty for no saving"
                 'start_epoch': 0,
                 'epochs': 301,
                 'fraction_warmup_steps': 0.01,  # type=float, help="Fraction of total number of steps"
                 'schedule': "step",
                 'warmup_steps': 10, # number of steps learning rate stays the same
                 # choices=("step", "multistep", "linear_with_warmup", "all_linear_with_warmup"
                 'eval_skip': 5,  # help='do evaluation every "eval_skip" frames'
                 'img_dir': 'Data/FromReports/mha/', 
                 'pretrained': False, #True,
                 'path_chansey': "/output/",
                 'device': "cuda",
                 'backbone_checkpoint': '',#'/output/Checkpoints_resnet50_1/checkpoint0019.pth',
                 'model_checkpoint': '',#'pretrained_Mdetr_resnet101_checkpoint.pth',
                 'logging_path': '/output/runs/detr_umc_20_mixed_new',
                 'save_freq': 5,
                 'dataset': 'umcmixed',
                 'umc_data_portion': 0.2
                 } 
    args = Namespace(**args_dict)
    main(args)
    #test_dataset(args)
    #test2(args)
    #compute_stats(args)
    #dataset_len(args)


#model([image.float()], ["this is a test"])
#print(targets)

"""
model, criterion, weight_dict = build_model(args)
#print(model)

x = NestedTensor.from_tensor_list([torch.randn(3, 224, 224)]) # random test input
targets = {}
targets['boxes'] = tensor([(0.5, 0.5, 0.1, 0.1)])
targets["tokens_positive"] = tensor([[(10, 14)]])
targets['labels'] = tensor([1])

y, cache = model(x, ["this is a test"])

print(y.keys())
print(cache.keys())
print(cache['tokenized'])
positive_map = create_positive_map(cache['tokenized'], targets["tokens_positive"])
#print(positive_map)
loss_dict = {}
loss_dict.update(criterion(y, [targets], positive_map))
print(loss_dict)
"""

"""
classes = ['A', 'B', 'C', 'D', 'E', 'N/A']

output = process_model_output(y, classes, (224, 224), confidence=0.0)

print(output)

#print_image_prediction(x.tensors[0].permute(1, 2, 0), output) # for plotting the color channel has to be the last

postprocessor = build_postprocessors()
results = postprocessor["bbox"](y, torch.Tensor([[224, 224]]))
res = {target: output for target, output in enumerate(results)}

print(res)
print()
print(y['pred_logits'])
print()
norm = nn.Softmax(-1)
out_prob = norm(y["pred_logits"].flatten(0, 1))
print(out_prob.unsqueeze(1)) # probability distribution
"""


