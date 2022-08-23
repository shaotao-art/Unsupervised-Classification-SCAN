import torch
import os
import pathlib
from torch.utils.data import DataLoader

from configs.config_simclr import CFG
from src_py.transforms import ContrastiveTransformations, SimclrT
from src_py.datasets import BaseData, SimclrData
from src_py.models import get_model
from src_py.losses import SimCLRLoss
from src_py.utils import adjust_learning_rate, get_logger, set_seed
from src_py.training_loop import simclr_train 


def main():
    pathlib.Path(CFG.log_fold).mkdir(parents=True, exist_ok=True) 

    # get logger
    logger = get_logger('simclr, texture', CFG.log_file_path)
    logger.info('getting logger...')  
    
    # set seed
    set_seed(2022)
    logger.info('setting seed to 2022...')

    # mkdir for ckps
    pathlib.Path(CFG.ckp_path).mkdir(parents=True, exist_ok=True) 
    logger.info(f'making dir for ckps: {CFG.ckp_path}')
    
    # data transform, dataset and dataloader
    T = SimclrT(CFG.img_size)
    Contrative_T = ContrastiveTransformations(T)
    base_dataset = BaseData(transform=None, train=True, download=True)
    dataset =  SimclrData(base_dataset, Contrative_T)
    data_loader = DataLoader(dataset, shuffle=True, batch_size=CFG.b_s, drop_last=True, num_workers=CFG.num_workers)
    logger.info(f'len dataset: {len(dataset)}')
    logger.info(f'batch_size: {CFG.b_s}, len dataloader: {len(data_loader)}')
    logger.info(f'train img_size: {CFG.img_size}')

    # getting model, optimizer and criterion
    model = get_model(CFG.model_config)
    optimizer = torch.optim.SGD(model.parameters(), lr=CFG.l_r, nesterov=False, weight_decay=0.0001, momentum=0.9)
    criterion = SimCLRLoss(CFG.temprature)
    logger.info(f'model : {model}')

    # loadding Checkpoint
    if os.path.exists(CFG.model_save_path):
        logger.info('ckp found, loading checkpoint.')
        checkpoint = torch.load(CFG.model_save_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.to(CFG.device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        logger.info('no ckp found, training from scratch')
        start_epoch = 0
        model = model.to(CFG.device)


    # training
    logger.info(f'starting training !')
    logger.info(f'num_epoch: {CFG.num_epoch}')
    logger.info(f'init learning_rate: {CFG.l_r}')
    logger.info(f'start epoch : {start_epoch}')
    for epoch in range(start_epoch, CFG.num_epoch):
            # Adjust lr
            lr = adjust_learning_rate(optimizer, epoch, CFG.num_epoch, CFG.l_r, CFG.min_lr)
            logger.info('Adjusted learning rate to {:.5f}'.format(lr))

            # Train
            logger.info('Train ...')
            simclr_train(data_loader, model, criterion, optimizer, epoch, CFG.device, logger)
        
            # Checkpoint
            logger.info('Checkpoint ...')
            torch.save({'optimizer': optimizer.state_dict(), 
                        'model': model.state_dict(), 
                        'epoch': epoch + 1}, 
                        CFG.model_save_path)
            
            if epoch % CFG.model_snap_shot_interval == 0:
                torch.save({'optimizer': optimizer.state_dict(), 
                            'model': model.state_dict(), 
                            'epoch': epoch + 1}, 
                            CFG.model_save_path+f'_ep{epoch}')


if __name__ == '__main__':
    main()