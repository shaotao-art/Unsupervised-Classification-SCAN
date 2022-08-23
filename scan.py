import os
import pathlib
import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.config_scan import CFG
from src_py.datasets import BaseData, ScanData
from src_py.transforms import ScanT
from src_py.models import get_model
from src_py.losses import SCANLoss
from src_py.utils import get_logger
from src_py.training_loop import scan_train
from src_py.utils import set_seed


def get_dataloader(knn_indices_path, topk, logger):
    indices = np.load(knn_indices_path)[:, :topk+1]
    dataset =  BaseData(transform=None, train=True, download=False)
    T = ScanT(CFG.img_size)
    scan_dataset = ScanData(dataset, indices, T)
    data_loader = DataLoader(scan_dataset, batch_size=CFG.b_s, shuffle=True, drop_last=True, num_workers=CFG.num_workers)
    logger.info(f'len dataset: {len(dataset)}')
    logger.info(f'batch_size: {CFG.b_s}, len dataloader: {len(data_loader)}')
    logger.info(f'train img_size: {CFG.img_size}')
    return data_loader

def main():
    pathlib.Path(CFG.log_fold).mkdir(parents=True, exist_ok=True) 
    # logger
    logger = get_logger('train scan', CFG.log_file_path)

    # set seed
    set_seed(2022)
    logger.info('setting seed to 2022...')

    # get data
    data_loader = get_dataloader(CFG.knn_indices_path, CFG.topk, logger)

    # get model, optimizer, criterion
    model = get_model(CFG.model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.l_r, weight_decay=0.0001)
    criterion = SCANLoss(entropy_weight=CFG.entropy_weight)
    logger.info(f'model: {model}')
    
    # Checkpoint
    if os.path.exists(CFG.model_save_path):
        logger.info('ckp found, loading ckp...')
        checkpoint = torch.load(CFG.model_save_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.to(CFG.device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        # optimizer.param_groups[0]['capturable'] = True
        start_epoch = checkpoint['epoch']
    else:
        logger.info('no ckp found, training from scratch')
        start_epoch = 0
        # load pretext model weights
        weights = torch.load(CFG.pretext_model_path, map_location='cpu')['model']
        backbone_weights = dict()
        for k, v in weights.items():
            if 'backbone' in k:
                backbone_weights[k[9:]] = v
        model.backbone.load_state_dict(backbone_weights)
        model = model.to(CFG.device)

    # training
    logger.info(f'starting training !')
    logger.info(f'num_epoch: {CFG.num_epoch}')
    logger.info(f'init learning_rate: {CFG.l_r}')
    logger.info(f'start epoch : {start_epoch}')
    for epoch in range(start_epoch, CFG.num_epoch):
            # Train
            logger.info('Train ...')
            scan_train(data_loader, model, criterion, optimizer, epoch, CFG.device, logger, CFG.update_head_only)
        
            # Checkpoint
            logger.info('Checkpoint ...')
            torch.save({'optimizer': optimizer.state_dict(), 
                        'model': model.state_dict(), 
                        'epoch': epoch + 1}, 
                        CFG.model_save_path)

            if epoch % CFG.model_snapshot_interval == 0:
                torch.save({'optimizer': optimizer.state_dict(), 
                            'model': model.state_dict(), 
                            'epoch': epoch + 1},
                            CFG.model_save_path+f'_epoch{epoch}')


if __name__ == '__main__':
    main()
