# Copyright (c) 2024 The Johns Hopkins University Applied Physics Laboratory

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Code for Stage 1 of UNITE: UMT pre-training on target domain data
"""

import argparse
import datetime
import time
import json
import math
import sys
import os
from pathlib import Path
from functools import partial
from typing import Iterable
from collections import OrderedDict

import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from timm.models import create_model
import numpy as np
import wandb

from src.optim_factory import create_optimizer, LayerDecayValueAssigner
from src.datasets import build_pretraining_dataset, build_dataset
from src.datasets.distributed import DistributedSampler
from src.utils import NativeScalerWithGradNormCount as NativeScaler
from src.utils import multiple_pretrain_samples_collate
from src import utils
from src.models import *

def get_args(args=None):
    parser = argparse.ArgumentParser('UMT Adaptation Script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--batch_size_val', default=64, type=int)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)
    parser.add_argument('--checkpoints_enabled', action='store_true')
    parser.set_defaults(checkpoints_enabled=True)
    parser.add_argument('--checkpoints_disabled', action='store_false', dest='checkpoints_enabled')

    # Model parameters
    parser.add_argument('--model', default='pretrain_umt_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--student_init', default='', type=str, help="Initialization weights for student model")
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--student_prefix', default='', type=str, help="Prefix for student model")
    parser.add_argument('--decoder_depth', default=4, type=int,
                        help='depth of decoder')
    parser.add_argument('--mask_type', default='attention', choices=['random', 'tube', 'attention'],
                        type=str, help='masked strategy of video tokens/patches')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size for backbone')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--normlize_target', default=True, type=bool,
                        help='normalized the target patch pixels')
    parser.add_argument('--tubelet_size', default=2, type=int,
                        help='temporal tube size for the patch embedding')
    parser.add_argument('--use_learnable_pos_emb', action='store_true')
    parser.set_defaults(use_learnable_pos_emb=False)
    parser.add_argument('--use_mean_pooling', action='store_false', dest='use_cls_token')
    parser.add_argument('--use_cls_token', action='store_true', dest='use_cls_token')
    parser.set_defaults(use_cls_token=True)

    # CLIP decoder parameters
    parser.add_argument('--clip_teacher', default='clip_b16', type=str,
                        help='Name of CLIP teacher')
    parser.add_argument('--clip_input_resolution', default=224, type=int,
                        help='input resolution of CLIP decoder')
    parser.add_argument('--clip_loss_ratio', default=1., type=float,
                        help='ratio for CLIP loss, pixel_loss + RATIO * clip_loss')
    parser.add_argument('--clip_loss_type', default='l2', type=str,
                        help='type of CLIP loss')
    parser.add_argument('--clip_loss_data', default='mixed', type=str)
    parser.add_argument('--clip_decoder_type', default='SA_Decoder', type=str,
                        help='type of CLIP decoder')
    parser.add_argument('--clip_decoder_embed_dim', default=512, type=int,
                        help='embedding dimension of CLIP decoder')
    parser.add_argument('--clip_output_dim', default=768, type=int,
                        help='output dimension of CLIP decoder')
    parser.add_argument('--clip_norm_type', default='l2', type=str,
                        help='type of feature normalization')
    parser.add_argument('--clip_return_attn', default=False, type=bool,
                        help='whether return CLIP attention')
    parser.add_argument('--clip_return_layers', default=[6,7,8,9,10,11], type=int, nargs='+',
                        help='list of CLIP layers to return')
    parser.add_argument('--clip_return_interval', default=1, type=float,
                        help='interval of CLIP teacher return layers')
    parser.add_argument('--clip_student_return_interval', default=1, type=float,
                        help='interval of CLIP student return layers')
    parser.add_argument('--clip_decoder_init')
    parser.add_argument('--freeze_clip_decoders', action='store_true', default=False)
    parser.add_argument('--no_freeze_clip_decoders', action='store_false', dest='freeze_clip_decoders')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--layer_decay', type=float, default=1.0,
                        help='Layer-wise decay factor (default: 1.0)')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.set_defaults(use_checkpoint=False)
    parser.add_argument('--checkpoint_num', type=int, default=0)

    # Augmentation parameters
    parser.add_argument('--num_sample', type=int, default=1, help='Repeated_aug (default: 1)')
    parser.add_argument('--color_jitter', type=float, default=0.0, metavar='PCT',
                        help='Color jitter factor (default: 0.0)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--flip', default=False,
                        help='whether flip the video in pretraining')

    # Dataset parameters
    parser.add_argument('--dataset', default='', type=str, help='name of domain shift dataset. ' 
                        'if this is specified, we will automatically override things like ' 
                        'ann file paths and number of classes.')
    parser.add_argument('--prefix', default='', type=str, help='prefix for data')
    parser.add_argument('--split', default=' ', type=str, help='split for metadata')
    parser.add_argument('--data_set', default='Kinetics_sparse', type=str, help='Dataset type')
    parser.add_argument('--train_fraction', default=1., type=float, help='fraction of training data')
    parser.add_argument('--train_repetitions', default=1, type=int, help='repetitions of train dataset')
    parser.add_argument('--ann_file_train', default=None, type=str, help='annotation path')
    parser.add_argument('--ann_file_train_target', default=None, type=str, help='annotation path')
    parser.add_argument('--ann_file_train_knn', default=None, type=str, help='annotation path')
    parser.add_argument('--nb_classes', default=400, type=int, help='number of classes')
    parser.add_argument('--ann_file_val', default=None, type=str, help='annotation path')
    parser.add_argument('--ann_file_test', default=None, type=str, help='annotation path')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--use_decord', default=True,
                        help='whether use decord to load video, otherwise load image')
    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--umt_step', type=int, default=1, help='controls the `new_step` parameter in mae dataset')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--test_best', action='store_true',
                        help='Whether test the best model')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # logging
    parser.add_argument('--log_freq', default=10, type=int)
    parser.add_argument('--disable_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_entity', type=str)
    parser.add_argument('--wandb_project', type=str)
    parser.add_argument('--output_dir', default='',
                    help='path where to save, empty for no saving')
    parser.add_argument('--wandb_group', default=None, type=str)

    # evaluation
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--test_num_segment', type=int, default=5)
    parser.add_argument('--test_num_crop', type=int, default=3)

    parser.add_argument('--val_interval', default=1, type=int, help='epoch interval for eval')
    parser.add_argument('--initial_validation', action='store_true', default=False)

    # YAML config
    parser.add_argument('--config', default='', type=str, help='yaml config file path')
    
    if args is not None:
        cmd_args = parser.parse_args(args)
    else:
        cmd_args = parser.parse_args()

    if cmd_args.config:
        # Get the configs from the yaml file
        yaml_args = argparse.Namespace()
        with open(cmd_args.config, 'r') as f:
            yaml_args.__dict__ = yaml.safe_load(f)
        # Overwrite yaml args with commandline args
        all_args = parser.parse_args(namespace=yaml_args)
    else:
        all_args = cmd_args

    if all_args.dataset:
        all_args = update_dataset_args_from_yaml(all_args)

    return all_args
    
def update_dataset_args_from_yaml(args):
    # See if yaml file path exists (dataset_mappings.yaml)
    dataset_mappings_path = os.path.join(os.path.dirname(__file__), 'dataset_mappings.yaml')
    if os.path.exists(dataset_mappings_path):
        with open(dataset_mappings_path, 'r') as f:
            dataset_mappings = yaml.safe_load(f)
        try:
            dataset_args = dataset_mappings[args.dataset]
            for k, v in dataset_args.items():
                setattr(args, k, v)
                print("Updated %s to %s" % (k, v))
        except KeyError:
            print(f"Dataset <{args.dataset}> not found in dataset_mappings.yaml")
            raise KeyError
    else:
        print("No dataset_mappings.yaml file found, skipping update_dataset_args_from_yaml!")
        raise FileNotFoundError

    return args


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        use_learnable_pos_emb=args.use_learnable_pos_emb,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_checkpoint=args.use_checkpoint,
        checkpoint_num=args.checkpoint_num,
        clip_decoder_embed_dim=args.clip_decoder_embed_dim,
        clip_output_dim=args.clip_output_dim,
        clip_norm_type=args.clip_norm_type,
        num_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        clip_return_layers=args.clip_return_layers,
        clip_student_return_interval=args.clip_student_return_interval,
        use_cls_token=args.use_cls_token,
    )
    return model

def train_one_epoch(
        model: torch.nn.Module, data_loader: Iterable, data_loader_train_target: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
        log_writer=None, lr_scheduler=None, start_steps=None,
        lr_schedule_values=None, wd_schedule_values=None, src_classifier=None,
        teacher_model=None, clip_input_resolution=224,
        clip_loss_type='l2', clip_loss_ratio=0.5,
        mask_type='tube', mask_ratio=0., use_wandb=False, args=None,
    ):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch [{}]:'.format(epoch)
    ipe = len(data_loader)
    print_freq = args.log_freq
    
    if data_loader_train_target is not None:
        data_loader_train_target_iter = iter(data_loader_train_target)
        def get_batch_target():
            nonlocal data_loader_train_target_iter
            try:
                return next(data_loader_train_target_iter)
            except StopIteration:
                data_loader_train_target_iter = iter(data_loader_train_target)
                return next(data_loader_train_target_iter)

    for step, batch in enumerate(metric_logger.log_every(
                                data_loader, print_freq, args.epochs, epoch, ipe, header=header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    try:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    except IndexError:
                        param_group["lr"] = lr_schedule_values[-1] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    try:
                        param_group["weight_decay"] = wd_schedule_values[it]
                    except IndexError:
                        param_group["weight_decay"] = wd_schedule_values[-1]

        # Get source domain batch
        videos, bool_masked_pos, labels_s = batch
        B_s = videos.shape[0]
        # Get target domain batch
        if data_loader_train_target is not None:
            videos_t, bool_masked_pos_t, labels_t = get_batch_target()
            videos = torch.cat([videos, videos_t], dim=0)
            bool_masked_pos = torch.cat([bool_masked_pos, bool_masked_pos_t], dim=0)
        
        videos = videos.to(device, non_blocking=True)
        labels_s = labels_s.to(device, non_blocking=True)
        if mask_type in ['attention']:
            bool_masked_pos = None
        else:
            bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        ###############################
        #### Teacher Model Forward ####
        ###############################

        with torch.no_grad():
            # calculate the predicted CLIP features
            B, C, T, H, W = videos.shape
            if H != clip_input_resolution:
                clip_videos = torch.nn.functional.interpolate(
                    videos.view(B, C*T, H, W), 
                    size=(clip_input_resolution, clip_input_resolution), 
                    mode='bicubic', align_corners=False
                )
                clip_videos = clip_videos.view(B, C, T, clip_input_resolution, clip_input_resolution)
            else:
                clip_videos = videos
            
            with torch.cuda.amp.autocast():
                if bool_masked_pos is None:
                    norm_clip, attn = teacher_model(clip_videos)
                else:
                    norm_clip = teacher_model(clip_videos)

            BT, N = attn.shape
            N_vis = N - int(N * mask_ratio)
            if mask_type == 'attention':
                importance = torch.multinomial(attn, N)
                bool_masked_pos = torch.ones((BT, N))
                pos1 = torch.arange(BT).view(-1, 1).repeat(1, N_vis)
                pos2 = importance[:, :N_vis]
                bool_masked_pos[pos1, pos2] = 0
                bool_masked_pos = bool_masked_pos.view(B, -1).to(torch.bool)
                    
            C_CLIP = norm_clip.shape[-1]
            if len(norm_clip.shape) == 4:
                K = norm_clip.shape[0]
                clip_bool_masked_pos = bool_masked_pos.unsqueeze(0).repeat(K, 1, 1)
                targets_clip_vis = norm_clip[~clip_bool_masked_pos].reshape(K, B, -1, C_CLIP)
            else:
                clip_bool_masked_pos = bool_masked_pos
                targets_clip_vis = norm_clip[~clip_bool_masked_pos].reshape(B, -1, C_CLIP)
            targets_clip = targets_clip_vis

        ###############################
        #### Student Model Forward ####
        ###############################

        if clip_loss_type == 'mse':
            loss_func_clip = nn.MSELoss()
        elif clip_loss_type == 'smooth_l1':
            loss_func_clip = nn.SmoothL1Loss()
        elif clip_loss_type == 'l1':
            loss_func_clip = nn.L1Loss()

        with torch.cuda.amp.autocast():
            # Pass the masked inputs through the model
            if src_classifier is not None:
                outputs_enc_masked, outputs_clip = model(videos, bool_masked_pos)
            else:
                outputs_clip = model(videos, bool_masked_pos, clip_only=True)

            # Select which data should participate in the CLIP loss
            if args.clip_loss_data == 'source':
                targets_clip = targets_clip[:, :B_s, :, :]
                outputs_clip = outputs_clip[:, :B_s, :, :]
            elif args.clip_loss_data == 'target':
                targets_clip = targets_clip[:, B_s:, :, :]
                outputs_clip = outputs_clip[:, B_s:, :, :]
            elif args.clip_loss_data == 'mixed':
                pass
            else:
                raise NotImplementedError
            
            # Compute CLIP alignment loss
            if clip_loss_type == 'l2':
                loss_clip = (2 - 2 * (outputs_clip * targets_clip).sum(dim=-1)).mean()
            elif clip_loss_type in ['mse', 'smooth_l1', 'l1']:
                loss_clip = loss_func_clip(input=outputs_clip, target=targets_clip)
            else:
                raise NotImplementedError

            # Compute total loss
            loss = loss_clip

        loss_clip = loss_clip.item()
        loss_value = loss.item() # total loss

        ######################
        #### Model Update ####
        ######################

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_clip=loss_clip)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_clip=loss_clip, head="loss_clip")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()
        
        if use_wandb and args.log_freq > 0 and step % args.log_freq == 0:
            wandb.log({
                "train/loss": loss_value,
                "train/loss_clip": loss_clip,
                "train/loss_scale": loss_scale_value,
                "train/lr": max_lr,
                "train/min_lr": min_lr,
                "train/weight_decay": weight_decay_value,
                "train/grad_norm": grad_norm,
                "train/epoch": epoch,
            })
        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    timestep = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestep}] Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    

def gather_predictions(predictions, device):
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    all_predictions = [torch.zeros_like(predictions) for _ in range(world_size)]
    torch.distributed.all_gather(all_predictions, predictions)

    if rank == 0:
        all_predictions = torch.cat(all_predictions, dim=0)
    return all_predictions

def load_student_from_ckpt(args, model):  
    print("Loading student model from %s" % args.student_init)
    checkpoint = torch.load(args.student_init, map_location='cpu')

    checkpoint_model = None
    for model_key in args.model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            checkpoint_model = {f'encoder.{k}': v for k, v in checkpoint_model.items()}

            all_keys = list(checkpoint_model.keys())
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
        all_keys = list(checkpoint_model.keys())

    new_dict = OrderedDict()
    for key in all_keys:
        if key.startswith('backbone.'):
            new_dict[key[9:]] = checkpoint_model[key]
        # elif key.startswith('encoder.'):
        #     new_dict[key[8:]] = checkpoint_model[key]
        else:
            new_dict[key] = checkpoint_model[key]
    checkpoint_model = new_dict

    # Add the decoders to the checkpoint
    if args.clip_decoder_init:
        decoder_ckpt = torch.load(args.clip_decoder_init, map_location='cpu')
        decoder_params = {k: v for k, v in decoder_ckpt.items() if k.startswith('clip_decoder.')}
        checkpoint_model.update(decoder_params)
        print("Loaded decoder params from %s" % args.clip_decoder_init)

    # interpolate position embedding
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
        num_patches = model.patch_embed.num_patches # 
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1

        # we use 8 frames for pretraining
        orig_t_size = 8 // model.patch_embed.tubelet_size
        new_t_size = args.num_frames // model.patch_embed.tubelet_size
        # height (== width) for the checkpoint position embedding
        orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(orig_t_size)) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int((num_patches // (new_t_size) )** 0.5)
        
        if orig_t_size != new_t_size:
            print(f"Temporal interpolate from {orig_t_size} to {new_t_size}")
            tmp_pos_embed = pos_embed_checkpoint.view(1, orig_t_size, -1, embedding_size)
            tmp_pos_embed = tmp_pos_embed.permute(0, 2, 3, 1).reshape(-1, embedding_size, orig_t_size)
            tmp_pos_embed = torch.nn.functional.interpolate(tmp_pos_embed, size=new_t_size, mode='linear')
            tmp_pos_embed = tmp_pos_embed.view(1, -1, embedding_size, new_t_size)
            tmp_pos_embed = tmp_pos_embed.permute(0, 3, 1, 2).reshape(1, -1, embedding_size)
            checkpoint_model['pos_embed'] = tmp_pos_embed
            pos_embed_checkpoint = tmp_pos_embed

        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # B, L, C -> BT, H, W, C -> BT, C, H, W
            pos_tokens = pos_tokens.reshape(-1, new_t_size, orig_size, orig_size, embedding_size)
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, new_t_size, new_size, new_size, embedding_size) 
            pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
    
    utils.load_state_dict(model, checkpoint_model, prefix=args.student_prefix)

    if args.freeze_clip_decoders:
        for name, param in model.named_parameters():
            if name.startswith('clip_decoder.'):
                param.requires_grad = False
                print("Freezing %s" % name)

    return model

def main(args):
    #######################
    #### General Setup ####
    #######################

    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_rank = global_rank
    if utils.is_main_process() and args.output_dir is not None:
        # Create output directory
        if utils.experiment_exists(args.output_dir):
            args.output_dir = utils.confirm_exp_overwrite(args.output_dir)    
        os.makedirs(args.output_dir, exist_ok=True)
            
        # Save args as a yaml file to output directory
        with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
            yaml.dump(vars(args), f, default_flow_style=False)
        print(args)
        log_writer = utils.TensorboardLogger(log_dir=args.output_dir)
    else:
        log_writer = None

    use_wandb = (utils.is_main_process() and
                not args.disable_wandb and
                'scrap' not in args.output_dir.lower()
    )
    if use_wandb:
        run_name = args.output_dir.split('/')[-1]
        if run_name == 'random':
            run_name = None
        wandb.init(entity=args.wandb_entity,
                   project=args.wandb_project,
                   group=args.wandb_group,
                   name=run_name,
                   config=args)

    cudnn.benchmark = True

    ##########################
    #### Data Preparation ####
    ##########################

    # Make datasets
    dataset_train = build_pretraining_dataset(args,
                                              args.ann_file_train,
                                              fraction=args.train_fraction)
    dataset_val, _ = build_dataset(is_train=False, test_mode=False, args=args)
    dataset_test, _ = build_dataset(is_train=False, test_mode=True, args=args)

    # Make samplers
    sampler_train = DistributedSampler(dataset_train,
                                        num_replicas=num_tasks,
                                        rank=sampler_rank,
                                        shuffle=True,
                                        repetitions=args.train_repetitions)
    sampler_val = torch.utils.data.DistributedSampler(dataset_val,
                                                      num_replicas=num_tasks,
                                                      rank=sampler_rank,
                                                      shuffle=False)
    sampler_test = torch.utils.data.DistributedSampler(dataset_test,
                                                       num_replicas=num_tasks,
                                                       rank=sampler_rank,
                                                       shuffle=False)
    print("Sampler_train = %s" % str(sampler_train))

    if args.num_sample > 1:
        collate_func = partial(multiple_pretrain_samples_collate, fold=False)
    else:
        collate_func = None

    # Create dataloaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_func,
        worker_init_fn=utils.seed_worker,
        persistent_workers=True
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size_val,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=True
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=True
    )

    # Configure target domain dataset, sampler and loader
    if args.ann_file_train_target: 
        dataset_train_target = build_pretraining_dataset(args, args.ann_file_train_target)
        if len(dataset_train_target) < len(dataset_train):
        # target dataset is smaller than source dataset
            target_repetitions = int(np.ceil(len(dataset_train) / len(dataset_train_target)))
            print("Repeating target dataset %d times" % target_repetitions)
        else:
            # target dataset is larger than source dataset, so we need to repeat train
            target_repetitions = 1
            train_repetitions = int(np.ceil(len(dataset_train_target) / len(dataset_train)))
            # train_repetitions = 1
            sampler_train = DistributedSampler(dataset_train,
                                                                num_replicas=num_tasks,
                                                                rank=sampler_rank,
                                                                shuffle=True,
                                                                repetitions=train_repetitions)
            data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            collate_fn=collate_func,
            worker_init_fn=utils.seed_worker,
            persistent_workers=True)
            print("Repeating source dataset %d times" % train_repetitions)

        sampler_train_target = DistributedSampler(dataset_train_target,
                                                repetitions=target_repetitions,
                                                num_replicas=num_tasks,
                                                rank=sampler_rank,
                                                shuffle=True)
        data_loader_train_target = torch.utils.data.DataLoader(
            dataset_train_target, sampler=sampler_train_target,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            collate_fn=collate_func,
            worker_init_fn=utils.seed_worker,
            persistent_workers=True
        )

    else:
        data_loader_train_target = None

    num_training_steps_per_epoch = len(data_loader_train) 
    
    ###########################
    #### Model Preparation ####
    ###########################

    # Student Model
    model = get_model(args)
    if args.student_init:
        model = load_student_from_ckpt(args, model)
        print("Loaded student model!")

    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    print("Tubelet size = %s" % str(args.tubelet_size))
    args.window_size = (args.num_frames // args.tubelet_size, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    model.to(device)
    model_without_ddp = model
    n_parameters = utils.count_parameters(model)
    print("Model = %s" % str(model))
    print('Student Params: {} M'.format(n_parameters / 1e6))

    # Teacher Model
    teacher_model = eval(args.clip_teacher)(
        clip_norm_type=args.clip_norm_type,
        input_resolution=args.clip_input_resolution,
        return_attn=args.clip_return_attn,
        clip_return_layers=args.clip_return_layers,
        clip_return_interval=args.clip_return_interval
    )
    teacher_model.to(device)

    print(f'Teacher model: {args.clip_teacher}')
    print('Teacher Params: {} M'.format(utils.count_parameters(teacher_model) / 1e6))
    print(f'Loss ratio: {args.clip_loss_ratio}')
    print(f'Loss type: {args.clip_loss_type}')

    total_batch_size = args.batch_size * utils.get_world_size() * (2 if data_loader_train_target is not None else 1)

    args.lr = args.lr * total_batch_size * args.num_sample / 256
    args.min_lr = args.min_lr * total_batch_size * args.num_sample / 256
    args.warmup_lr = args.warmup_lr * total_batch_size * args.num_sample / 256
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Repeated sample = %d" % args.num_sample)
    print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    num_layers = 12 # TODO: make this configurable
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=(False))
        model_without_ddp = model.module
        teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.gpu], find_unused_parameters=False)

    ##################################
    #### Optimization Preparation ####
    ##################################

    # Layer-wise LR decay
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
        print("Assigned values = %s" % str(assigner.values))
    else:
        assigner = None

    skip_weight_decay_list = model_without_ddp.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)

    optimizer = create_optimizer(args, model_without_ddp, skip_list=skip_weight_decay_list,
                                 get_num_layer=assigner.get_layer_id if assigner is not None else None,
                                 get_layer_scale=assigner.get_scale if assigner is not None else None)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    ##################
    #### Training ####
    ##################

    if args.auto_resume:
        utils.auto_load_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    torch.cuda.empty_cache()
    print(f"Start training for {args.epochs} epochs")
    print(f"Mask ratio: {args.mask_ratio}")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(
            model, data_loader_train, data_loader_train_target,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            src_classifier=None,
            teacher_model=teacher_model, 
            clip_input_resolution=args.clip_input_resolution,
            clip_loss_type=args.clip_loss_type, 
            clip_loss_ratio=args.clip_loss_ratio,
            mask_type=args.mask_type,
            mask_ratio=args.mask_ratio,
            use_wandb=use_wandb,
            args=args,
        )

        #################################
        #### Checkpointing & Logging ####
        #################################

        if args.output_dir and args.checkpoints_enabled and utils.is_main_process():
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, 
                    optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch
                )
            utils.save_latest_model(
                args=args, model=model, model_without_ddp=model_without_ddp, 
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch
            )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if use_wandb:
            wandb.log({'epoch': epoch})

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if use_wandb:
        wandb.finish()
     
if __name__ == '__main__':    
    opts = get_args()
    main(opts)
