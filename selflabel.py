import torch
from torch.utils.data import DataLoader
import os
import pathlib

from src_py.transforms import ScanT, ValT
from src_py.models import get_model
from src_py.losses import ConfidenceBasedCE
from src_py.utils import get_logger, set_seed, EMA
from src_py.training_loop import selflabel_train
from src_py.datasets import BaseData, SelfLabelData
from configs.config_selflabel import CFG


def main():
    pathlib.Path(CFG.log_fold).mkdir(parents=True, exist_ok=True) 
    # get logger
    logger = get_logger('selflabel', CFG.log_file_path)

    # set seed
    set_seed(2022)
    logger.info('setting seed to 2022...')

    # make dataloader
    weak_T = ValT(CFG.img_size)
    strong_T =ScanT(CFG.img_size)
    base_dataset =  BaseData(transform=None, train=True, download=False)
    dataset =  SelfLabelData(base_dataset, weak_T, strong_T)
    data_loader = DataLoader(dataset, shuffle=True, batch_size=CFG.b_s, drop_last=True, num_workers=CFG.num_workers)
    logger.info(f'len dataset: {len(dataset)}')
    logger.info(f'batch_size: {CFG.b_s}, len dataloader: {len(data_loader)}')
    logger.info(f'train img_size: {CFG.img_size}')


    model = get_model(CFG.model_config)
    criterion = ConfidenceBasedCE(CFG.confident_thres, CFG.apply_class_balancing)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.l_r, weight_decay=1e-4)

    # Checkpoint
    if os.path.exists(CFG.model_save_path):
        print('ckp found, loading ckp...')
        checkpoint = torch.load(CFG.model_save_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.to(CFG.device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        # optimizer.param_groups[0]['capturable'] = True
        start_epoch = checkpoint['epoch']
    else:
        print('no ckp found, training from scratch')
        start_epoch = 0
        # load scan model weights
        weights = torch.load(CFG.scan_model_path, map_location='cpu')['model']
        model.load_state_dict(weights)
        model = model.to(CFG.device)

    # EMA
    if CFG.ema:
        ema = EMA(model)
    else:
        ema = None

    # training
    logger.info(f'starting training !')
    logger.info(f'num_epoch: {CFG.num_epoch}')
    logger.info(f'init learning_rate: {CFG.l_r}')
    logger.info(f'start epoch : {start_epoch}')
    for epoch in range(start_epoch, CFG.num_epoch):
            # Train
            print('Train ...')
            selflabel_train(data_loader, model, criterion, optimizer, epoch, CFG.device, ema, logger)
        
            # Checkpoint
            print('Checkpoint ...')
            torch.save({'optimizer': optimizer.state_dict(), 
                        'model': model.state_dict(), 
                        'epoch': epoch + 1}, CFG.model_save_path)
            if epoch % CFG.model_snap_shot_interval == 0:
                torch.save({'optimizer': optimizer.state_dict(), 
                            'model': model.state_dict(), 
                            'epoch': epoch + 1}, CFG.model_save_path)


if __name__ == '__main__':
    main()