import enum
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import os
import os.path as osp
import json

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

        self.train_metrics = {'train/lr': 0, 'train/loss': 0, 'train/loss_classifier': 0, 'train/loss_box_reg': 0, 'train/loss_mask': 0, 'train/loss_objectness': 0, 'train/loss_rpn_box_reg': 0, 'train/time': 0, 'train/data': 0, 'train/max mem': 0}
        
        self.val_metrics = {}
        for task in tasks:
            list_metrics = ['val/' + task + '/Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]', 'val/' + task + '/Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]', 'val/' + task + '/Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]', 
                'val/' + task + '/Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]', 'val/' + task + '/Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]', 'val/' + task + '/Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]', 
                'val/' + task + '/Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]', 'val/' + task + '/Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]', 'val/' + task + '/Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]', 
                'val/' + task + '/Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]', 'val/' + task + '/Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]', 'val/' + task + '/Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]' ]
            
            d = {}
            for metric in list_metrics:
                d[metric] = 0  
            
            self.val_metrics[task] = d

            print('val_metrics' , self.val_metrics)
            


            




    def _train_step(self, e) -> None:

        #self.model.train()
        metric_logger =  train_one_epoch(self.model, self.optimizer, self.train_data_loader, self.device, e, train_steps = self.train_steps, print_freq = 10)
        for key in metric_logger.meters.keys():
            self.train_metrics['train/' + key] = metric_logger.meters[key].value
            



    def _val_step(self) -> None:

        #self.model.eval()
        coco_evaluator = evaluate(self.model, self.val_data_loader, device=self.device, val_steps=self.val_steps)

        for task in self.tasks:
            metrics = coco_evaluator.coco_eval[task].stats

            for i, key in enumerate(self.val_metrics.keys()):
                self.val_metrics[task][key] = metrics[i]
    

    def _write_metrics(self, epoch: int) -> None: #TODO to adapt : adapt metrics

        print('.... Saving metrics to tensorboard')

        for key, item in self.train_metrics.items():
            self.w.add_scalar(key, item, epoch)

        for task in self.tasks:
            for key, item in self.val_metrics[task].items():
                self.w.add_scalar(key, item, epoch)

        for name, param in self.model.named_parameters():
            self.w.add_histogram(name, param, epoch)
        print('done;')
        print()