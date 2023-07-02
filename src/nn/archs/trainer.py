import enum
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import os
import os.path as osp
import json

from src.tools.timer import timeit
from src.tools.base_trainer import BaseTrainer

from src.nn.references.detection.engine import train_one_epoch, evaluate







# ********************* trainer *********************

class Trainer(BaseTrainer):

    def __init__(
            self,
            device,
            distributed_training,
            model,
            tasks,
            epochs,
            batch_size,
            loss_fn,
            optimizer,
            lr_scheduler,
            patience,
            train_data_loader,
            train_steps,
            val_data_loader,
            val_steps,
            checkpoint_frequency,
            model_name,
            weights_path
    ) -> None:

        super().__init__(
            device,
            distributed_training,
            model,
            epochs,
            batch_size,
            loss_fn,
            optimizer,
            lr_scheduler,
            patience,
            train_data_loader,
            train_steps,
            val_data_loader,
            val_steps,
            checkpoint_frequency,
            model_name,
            weights_path,
        )
        self.tasks = tasks 

        self.coco = None

        self.train_metrics = {'train/lr': 0, 'train/loss': 0, 'train/loss_classifier': 0, 'train/loss_box_reg': 0, 'train/loss_mask': 0, 'train/loss_objectness': 0, 'train/loss_rpn_box_reg': 0, 'train/time': 0, 'train/data': 0, 'train/max mem': 0}
        
        self.val_metrics = {}

        # metrics
        for task in tasks:
            d = {}
            list_metrics = ['val/' + task + '/Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]', 'val/' + task + '/Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]', 'val/' + task + '/Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]', 
                'val/' + task + '/Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]', 'val/' + task + '/Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]', 'val/' + task + '/Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]', 
                'val/' + task + '/Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]', 'val/' + task + '/Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]', 'val/' + task + '/Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]', 
                'val/' + task + '/Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]', 'val/' + task + '/Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]', 'val/' + task + '/Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]' ]
            for metric in list_metrics:
                d[metric] = 0  
            self.val_metrics[task] = d

        # losses
        d = {}
        self.list_losses = ['loss', 'loss_classifier', 'loss_box_reg', 'loss_mask']
        for loss in self.list_losses:
                d['val/' + loss] = 0  
        d['val/' + 'evaluator_time'] = 0 
        d['val/' + 'model_time'] = 0

        self.val_metrics['losses'] = d




    def _train_step(self, e) -> None:

        #self.model.train()
        metric_logger =  train_one_epoch(self.model, self.optimizer, self.train_data_loader, self.device, e, train_steps = self.train_steps, print_freq = 10)
        for key in metric_logger.meters.keys():
            self.train_metrics['train/' + key] = metric_logger.meters[key].value
            



    def _val_step(self) -> None:

        #self.model.eval()
        coco_evaluator, self.coco, metric_logger = evaluate(self.model, self.val_data_loader, device=self.device, tasks = self.tasks, val_steps=self.val_steps, coco = self.coco)

        for task in self.tasks:
            metrics = coco_evaluator.coco_eval[task].stats

            for i, key in enumerate(self.val_metrics[task].keys()):
                self.val_metrics[task][key] = metrics[i]
        
        for key in metric_logger.meters.keys():
            self.val_metrics['losses']['val/' + key] = metric_logger.meters[key].value
    
    @timeit
    def _write_metrics(self, epoch: int) -> None: #TODO to adapt : adapt metrics

        print('.... Saving metrics to tensorboard')

        for key, item in self.train_metrics.items():
            self.w.add_scalar(key, item, epoch)

        for task in self.tasks:
            for key, item in self.val_metrics[task].items():
                self.w.add_scalar(key, item, epoch)
        for key, item in self.val_metrics['losses'].items():
                self.w.add_scalar(key, item, epoch)

        for loss in self.list_losses:
            self.w.add_scalars('train_val/' + loss, {'train_loss': self.train_metrics['train/' + loss],
                                                    'val_loss': self.val_metrics['losses']['val/' + loss]}, epoch)

        for name, param in self.model.named_parameters():
            self.w.add_histogram(name, param, epoch)

        print('done;')
        print()