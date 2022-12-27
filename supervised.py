import math
import sys
from typing import Iterable
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy 
import utils
from timm.utils import accuracy

def train_one_epoch(model: torch.nn.Module, args, train_config,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, amp_autocast,
                    device: torch.device, epoch: int, loss_scaler, 
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, model_ema=None):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    #metric_logger.add_meter('pseudo_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for step, ((images_weak, images_strong, mask), targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None: 
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]

        # ramp-up ema decay 
        # model_ema.decay = train_config['model_ema_decay_init'] + (args.model_ema_decay - train_config['model_ema_decay_init']) * min(1, it/train_config['warm_it'])
        # metric_logger.update(ema_decay=model_ema.decay)
        
        images_weak, images_strong = images_weak.to(device, non_blocking=True), images_strong.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True) 
        targets = targets.to(device, non_blocking=True)

     
            
            
        with amp_autocast(): 
            logits = model(images_strong)   
           
        loss = F.cross_entropy(logits,targets)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        
        if loss_scaler is not None:
            grad_norm = loss_scaler(loss, optimizer, clip_grad=None, parameters=model.parameters(), create_graph=False)
            loss_scale_value = loss_scaler.state_dict()["scale"]
            metric_logger.update(loss_scale=loss_scale_value)
            metric_logger.update(grad_norm=grad_norm)
        else:                   
            loss.backward(create_graph=False)       
            optimizer.step()

       
        torch.cuda.synchronize()  

        metric_logger.update(train_loss=loss_value)
       
        
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

        if log_writer is not None:  
            try: 
                log_writer.update(train_loss=loss_value, head="train")
            except OSError:
                continue 
            # try:
            #     log_writer.update(pseudo_loss=l.item(),head="train")
            #     print('Logged pseudo loss')
            # except OSError:
            #     continue 
            # try:
            #     log_writer.update(full_training_acc=full_pseudo_label_acc,head="train")
            #     print('Logged full training acc')
            # except OSError:
            #     continue 
            # try:          
            #     log_writer.update(loss_st=loss_st.item(), head="train")
            
            # except OSError:
            #     continue 
            # try:
            #     log_writer.update(loss_fair=loss_fair.item(), head="train")
            # except OSError:
            #     continue 
            
            # if args.mask:
            #     try:
            #         log_writer.update(loss_mim=loss_mim.item(), head="train")
            #     except OSError:
            #         continue 
            #     try:
            #         log_writer.update(loss_align=loss_align.item(), head="train")
            #     except OSError:
            #         continue 
            # try: 
            #     log_writer.update(conf_ratio=conf_ratio, head="train")
            # except OSError:
            #     continue 
            # try:
            #     log_writer.update(pseudo_label_acc=pseudo_label_acc, head="train")       
            # except OSError:
            #     continue 
            # try:   
            #     log_writer.update(lr=max_lr, head="opt")
            # except OSError:
            #     continue 
            # try: 
            #     log_writer.update(min_lr=min_lr, head="opt")
            # except OSError:
            #     continue 
            try:
                log_writer.set_step()
            except OSError:
                continue 

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
            

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(data_loader, model, device, model_ema=None, args=None):
    
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    model_ema = None 
    if model_ema is not None:
        model_ema.ema.eval()   
        
    if args.dataset in ['pets', 'caltech101']:
        all_outputs = []
        all_ema_outputs = []
        all_targets = []
        
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0].to(device, non_blocking=True)
        target = batch[-1].to(device, non_blocking=True)

        # compute output
        output = model(images)
            
        if args.dataset in ['pets', 'caltech101']:
            all_outputs.append(output.cpu())
            all_targets.append(target.cpu())   
        else:    
            acc = accuracy(output, target)[0]
            metric_logger.meters['acc1'].update(acc.item(), n=images.shape[0])
        
        if model_ema is not None:
            ema_output = model_ema.ema(images) 
            
            if args.dataset in ['pets', 'caltech101']:
                all_ema_outputs.append(ema_output.cpu())
            else:  
                ema_acc1 = accuracy(ema_output, target)[0]  
                metric_logger.meters['ema_acc1'].update(ema_acc1.item(), n=images.shape[0])

    if args.dataset in ['pets', 'caltech101']:
        mean_per_class = utils.mean_per_class(torch.cat(all_outputs), torch.cat(all_targets))
        metric_logger.meters['acc1'].update(mean_per_class) 
        if model_ema is not None:
            mean_per_class = utils.mean_per_class(torch.cat(all_ema_outputs), torch.cat(all_targets))
            metric_logger.meters['ema_acc1'].update(mean_per_class) 
            
    print('* Acc@1 {top1.global_avg:.3f}'.format(top1=metric_logger.acc1))    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

