from argparse import Namespace
from backbone import build_backbone
from data import DeepLesion, LesionData, UnmatchedLesionData
import torch
from torch.utils.data import DataLoader, ConcatDataset
import math
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Iterable
from metrics import MetricLogger, SmoothedValue, adjust_learning_rate
import time
import datetime
import utils
import os
from os import listdir
from os.path import isfile, join



def train_one_epoch(
    model: torch.nn.Module,
    criterion: Optional[torch.nn.Module],
    weight_dict: Dict[str, float],
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args,
    device: torch.device,
    max_norm: float = 0,
):
    model.train()
    if criterion is not None:
        criterion.train()
    metric_logger = MetricLogger(delimiter="  ", path=args.logging_path)
    metric_logger.add_meter("lr_backbone", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 100

    num_training_steps = int(len(data_loader) * args.epochs)
    for i, batch_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        curr_step = epoch * len(data_loader) + i

        samples = batch_dict["samples"].to(device)
        targets = batch_dict["targets"]
        targets = utils.targets_to(targets, device)

        out = model(samples)
        outputs = {"pred_boxes": out}

        loss_dict = {}
        if criterion is not None:
            loss_dict.update(criterion(outputs, targets))

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

        adjust_learning_rate(
            optimizer,
            epoch,
            curr_step,
            num_training_steps=num_training_steps,
            args=args,
        )

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr_backbone=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    print("Averaged stats:", metric_logger)
    metric_logger.updateTensorboard(epoch)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: Optional[torch.nn.Module],
    weight_dict: Dict[str, float],
    data_loader,
    device: torch.device,
    epoch,
    args
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

        out = model(samples)
        outputs = {"pred_boxes": out}

        loss_dict = {}
        if criterion is not None:
            loss_dict.update(criterion(outputs, targets))

        loss_dict_reduced = loss_dict
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )

    print("Averaged stats:", metric_logger)
    metric_logger.updateTensorboard(epoch, mode='test')


def save_checkpoint(output_dir, epoch, model, optimizer, save_freq=2):
    if (epoch + 1) % save_freq == 0:
        checkpoint_paths = output_dir + f"checkpoint{epoch:04}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_paths)


def main(args):
    path_chansey = '/output/'
    device = torch.device(args.device)
    model, criterion, weight_dict = build_backbone(args)
    #print(model)
    model.to(device)
    criterion.to(device)
    #train_dataset = DeepLesion(path_chansey + 'Pretraining/', path_chansey + 'Data/volumes/')
    #val_dataset = DeepLesion(path_chansey + 'Pretraining/', path_chansey + 'Data/volumes/', split='val')
    #train_sampler = torch.utils.data.RandomSampler(train_dataset)
    #val_sampler = torch.utils.data.RandomSampler(val_dataset)
    #batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=False)
    #batch_sampler_val = torch.utils.data.BatchSampler(val_sampler, args.batch_size, drop_last=False)
    #train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn)
    #eval_data_loader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, collate_fn=utils.collate_fn)
    
    
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
    
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    val_sampler = torch.utils.data.RandomSampler(val_dataset)
    batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=False)
    batch_sampler_val = torch.utils.data.BatchSampler(val_sampler, args.batch_size, drop_last=False)
    
    
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn)
    eval_data_loader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, collate_fn=utils.collate_fn)

    #img, target = train_dataset[31946]
    #print(target)

    param_dicts = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
            ]
        }
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr_backbone, weight_decay=args.weight_decay)
    
    if args.finetunecheckpoint:
        checkpoint = torch.load(args.path_chansey + args.finetunecheckpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("loaded checkpoint ", args.finetunecheckpoint, " for finetuning")
    
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
    

    print("Start training")
    start_time = time.time()
    best_metric = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        print(f"Starting epoch {epoch}")
        train_stats = train_one_epoch(model, criterion, weight_dict, train_dataloader, optimizer, epoch, args, device)

        if epoch % args.eval_skip == 0:
            print("Evaluating...")
            curr_test_stats = evaluate(model, criterion, weight_dict, eval_data_loader, device, epoch, args)
            print("test image: ")
            img, target = train_dataset[2]
            img = img.to(device)
            model.eval()
            out = model(img.unsqueeze(0))
            print(out[-1])

        if args.output_dir:
            save_checkpoint(path_chansey + args.output_dir, epoch, model, optimizer)


    if args.output_dir:
        #save_checkpoint(path_chansey + args.output_dir, epoch, model, optimizer)
        output_dir = path_chansey + args.output_dir
        checkpoint_paths = output_dir + f"checkpoint{epoch:04}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_paths)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    #img, target = train_dataset[32559]
    #model.eval()
    #out = model(img.unsqueeze(0))
    #print(out[-1])

def compute_stats(args):
    path_chansey = '/output/'
    dataset = DeepLesion(path_chansey + 'Pretraining/', path_chansey + 'Data/volumes/')
    loader = DataLoader(dataset,
                         batch_size=10,
                         num_workers=0,
                         shuffle=False, collate_fn=utils.collate_fn)

    mean = 0.
    std = 0.
    for batch_dict in loader:
        #print(batch_dict)
        images = batch_dict["samples"]
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    
    print("mean: ", mean)
    print("std: ", std)

if __name__ == '__main__':
    args_dict = {'lr_backbone': 1e-5,
                 'num_channels': 512,
                 'hidden_dim': 256,
                 'batch_size': 4, #2,
                 'schedule': "lambda", #"all_linear_with_warmup",
                 'lr_drop': 10,
                 'weight_decay': 5e-5,
                 'set_cost_bbox': 5,
                 'set_cost_giou': 2,
                 'bbox_loss_coef': 5,
                 'giou_loss_coef': 1,
                 'mse_loss_coef': 0,
                 'start_epoch': 0,
                 'epochs': 51,
                 'output_dir': 'Final_Backbone_Pretraining/resnet101_umc_only/',
                 'fraction_warmup_steps': 0.01,
                 'eval_skip': 5,
                 'device': "cuda",
                 'path_chansey': "/output/",
                 'logging_path': '/output/runs/resnet101_umc_only',
                 'img_back_pretrained': True, 
                 'backbone_type': 'resnet101',
                 'dataset': 'umc',
                 'umc_data_portion': 1,
                 'finetunecheckpoint': ''
                 }
    args = Namespace(**args_dict)
    main(args)
    #compute_stats(args)
