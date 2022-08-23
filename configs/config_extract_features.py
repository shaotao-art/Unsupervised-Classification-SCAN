import torch
import os

class CFG:
    # pretext model path
    ckp_path = './ckp'
    pretext_model_path = os.path.join(ckp_path, 'pretext.ckp')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # model
    model_config = {'mode': 'contrastive', 'feature_dim': 128}

    # only extract training set feature for latter training
    extract_train_feats = True
    extract_test_feats = True
    img_size = 32 # remember the pretext model's training image size should be align to this image size
    crop_num = 1 # take crop_num same images' features' mean

    # file save path
    extract_file_path = './extracted_files'
    train_save_path = {
        'npy': os.path.join(extract_file_path, 'train_features.npy'),
        'knn': os.path.join(extract_file_path, 'train_knn_indices.npy'),
    }

    test_save_path = {
        'npy': os.path.join(extract_file_path, 'test_features.npy'),
        'knn': os.path.join(extract_file_path, 'test_knn_indices.npy'),
    }
