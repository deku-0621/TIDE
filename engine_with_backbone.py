# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import torch
import DETR_util.misc as utils

def train_one_epoch_only(model: torch.nn.Module, criterion: torch.nn.Module,
                         data_loader: Iterable, optimizer: torch.optim.Optimizer,
                         device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    #metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_temp = 2
    #for querys, supports, querys_wh_tensor,supports_wh_tensor,targets in metric_logger.log_every(data_loader, print_freq, header):
    for sample_query, sample_support, targets in metric_logger.log_every(data_loader, print_freq, header):
        sample_query = sample_query.to(device)
        sample_support = sample_support.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]#ZladWu target: {'bbox':tag_bboxes,'labels':tag_labels}
        outputs = model(sample_query, sample_support) #ZladWu todo outputs:
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        #for name, param in model.named_parameters():
        #    if param.grad is None:
        #        print(name)


        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        #metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)

        loss_dict = {
            'loss_ce':loss_dict_reduced_scaled['loss_ce'],
            'loss_bbox':loss_dict_reduced_scaled['loss_bbox'],
            'loss_giou':loss_dict_reduced_scaled['loss_giou']
        }

        metric_logger.update(loss=loss_value, **loss_dict)

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if loss_value < loss_temp:
            loss_temp = loss_value
            torch.save(model.state_dict(), f'train_output/best_loss_{loss_value}.pth')
        #防止内存泄漏
        del sample_query, sample_support, targets, outputs, loss_dict, loss_dict_reduced, loss_dict_reduced_unscaled, loss_dict_reduced_scaled, losses_reduced_scaled, losses, loss_value

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


