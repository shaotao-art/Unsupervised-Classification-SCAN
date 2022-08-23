import torch
import os

class CFG:
    # model config
    model_config = {'mode': 'contrastive', 'feature_dim': 128}

    # hyperparameters
    debug = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epoch = 500
    img_size = 32
    b_s = 8 if debug == True else 512
    l_r = 0.4
    min_lr = 1e-4 # init and final learning rate for cosine learning rate schec
    num_workers = 4
    temprature = 0.1

    # model and log save path
    ckp_path = './ckp'
    log_fold = './logs'
    model_save_path = os.path.join(ckp_path, 'pretext.ckp')    
    log_file_path = os.path.join(log_fold, 'pretext.log')
    model_snap_shot_interval = 100
