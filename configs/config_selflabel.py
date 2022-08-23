import torch
import os

class CFG:
    # hyper parameters
    debug = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epoch = 200
    b_s = 16 if debug == True else 1000
    img_size = 32
    num_class = 10
    l_r = 1e-4
    num_head = 1
    model_config = {'mode':'scan', 'num_class':num_class, 'num_head': num_head}
    num_workers = 2

    # scan model path
    ckp_path = './ckp'
    log_fold = './logs/'
    scan_model_path = os.path.join(ckp_path, f'scan_{num_class}.ckp')
    model_save_path = os.path.join(ckp_path, f'selflabel_{num_class}.ckp')
    log_file_path = os.path.join(log_fold, 'self-label.log')
    model_snap_shot_interval = 20

    ema = False
    confident_thres = 0.99
    apply_class_balancing = True