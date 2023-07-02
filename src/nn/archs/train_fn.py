import torch
import torchvision
import numpy as np
import os.path as osp

from src.nn.archs.data_load import *
from src.nn.archs.helper import *
#from src.nn.archs.segmentation import *
from src.nn.archs.trainer import Trainer

from src.nn.models.models import *

import src.nn.references.detection.utils as utils

from src.tools.helper import log_config



torch.cuda.empty_cache()





def main(config):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device(config['device'])
    # use our dataset and defined transformations
    if config['dataset'] == 'pennpudanped':
        dataset = PennFudanDataset(config['train_ds'], get_transform(train=True))
        dataset_test = PennFudanDataset(config['val_ds'], get_transform(train=False))
        
    if config['dataset'] == 'endovis':
        dataset = EndovisTestDataset(config['train_ds'], get_transform(train=True))
        dataset_test = EndovisTestDataset(config['val_ds'], get_transform(train=False))
    
    if config['dataset'] == 'endovis_tuned':
        dataset = EndovisTunedDataset(config['train_ds'], get_transform(train=True), num_classes=config['num_classes'], train = True, val_frac = config['val_frac'])
        dataset_test = EndovisTunedDataset(config['val_ds'], get_transform(train=True), num_classes=config['num_classes'], train = False, val_frac = config['val_frac'])


    #removing bad samples from test ds
    indices = list(range(270))
    indices_bis = list(range(315, 360))
    idx = indices + indices_bis
    dataset_test = torch.utils.data.Subset(dataset_test, idx)


    # define training and validation data loaders
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    val_dataloader = torch.utils.data.DataLoader(
        dataset_test, batch_size=config['batch_size'], shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    nn = get_model_instance(config['num_classes'], config['hidden_layer_segm'] ,config['heads'])
    #nn = get_model_instance_segmentation(2)

    # # move model to the right device
    # model.to(device)

    # construct an optimizer
    params = [p for p in nn.parameters() if p.requires_grad]

    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, lr=config['lr'],
                                 weight_decay=0.0005)

    elif config['optimizer'] == 'sgd' :
        optimizer = torch.optim.SGD(params, lr=config['lr'], momentum=0.9,
                                 weight_decay=0.0005)
    # and a learning rate scheduler
    if 'lr_scheduler' in config:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=3,
                                                    gamma=0.1)

    else:
        lr_scheduler = None

    trainer = Trainer(
            device = device,
            distributed_training = config['distributed_training'],
            model = nn,
            tasks = config['tasks'],
            epochs = config['epochs'],
            batch_size = config['batch_size'],
            loss_fn = config['loss_fn'],
            optimizer = optimizer,
            lr_scheduler = lr_scheduler,
            patience = config['patience'],
            train_data_loader = train_dataloader,
            train_steps = config['train_steps'],
            val_data_loader = val_dataloader,
            val_steps = config['val_steps'],
            checkpoint_frequency = config['checkpoint_frequency'],
            model_name = config['model_name'],
            weights_path = config['weights_path'],
            )

    log_config(config, osp.join(config['weights_path'], config['model_name']))
    trainer.train()
    trainer.save_model()
    #trainer.classification_report()
    trainer.save_loss()


    print('NN saved to directory: ', config['weights_path'])
