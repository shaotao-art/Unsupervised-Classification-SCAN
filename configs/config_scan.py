import torch
import os

class CFG:
    # path to load pre-extract knn indices and names_lst
    extract_file_path = './extracted_files'
    knn_indices_path = os.path.join(extract_file_path, 'train_knn_indices.npy')

    # hyper parameters
    debug = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epoch = 50
    b_s = 128 if debug == False else 8
    l_r = 1e-4
    update_head_only = False

    # overcluster is very important for this dataset
    num_class = 10
    img_size = 32
    topk = 20
    num_head = 1
    model_config = {'mode':'scan', 'num_class':num_class, 'num_head':num_head}
    num_workers = 2
    entropy_weight = 5

    # pretext model path
    ckp_path = './ckp'
    log_fold = './logs/'
    pretext_model_path = os.path.join(ckp_path, 'pretext.ckp')
    # model save path
    model_save_path = os.path.join(ckp_path, f'scan_{num_class}.ckp')
    log_file_path = os.path.join(log_fold, f'scan_{num_class}.log')
    model_snapshot_interval = 20
