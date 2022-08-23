import faiss
import torch
from torch.utils.data import DataLoader
import pathlib
from tqdm import tqdm
import numpy as np

from src_py.transforms import ValT
from src_py.datasets import BaseData
from src_py.models import get_model
from configs.config_extract_features import CFG


def get_all_features(dataloader, model, device):
    feature_lst = []
    model = model.to(device)
    with torch.no_grad():
        for x in tqdm(dataloader):
            # extract the final output as features
            feature = model(x.to(device))
            feature_lst.append(feature)
    return torch.cat(feature_lst).cpu().numpy()
    

def mine_nearest_neighbors(features, topk=20):
    # mine the topk nearest neighbors for every sample
    n, dim = features.shape[0], features.shape[1]
    index = faiss.IndexFlatIP(dim)
    # index = faiss.index_cpu_to_all_gpus(index)
    index.add(features)
    distances, indices = index.search(features, topk+1) # Sample itself is included  
    return indices



def main():
    # mkdir for ckps
    pathlib.Path(CFG.extract_file_path).mkdir(parents=True, exist_ok=True) 
    print(f'making dir for extract files: {CFG.extract_file_path}')    

    # get model
    model = get_model(CFG.model_config)
    print(CFG.pretext_model_path)
    print(f'getting mode:', model)
    model.load_state_dict(torch.load(CFG.pretext_model_path,  map_location=torch.device(CFG.device))['model'])
    print('loading model weigths')


    if CFG.extract_train_feats:
        print('starting extract training set features')
        T = ValT(CFG.img_size)
        train_dataset =  BaseData(transform=T, train=True, download=False)
        train_data_loader = DataLoader(train_dataset, shuffle=False, batch_size=16, drop_last=False)
        print(f'len train data: {len(train_dataset)}')
        train_features = get_all_features(train_data_loader, model, CFG.device)
        print('train features shape:', train_features.shape)
        np.save(CFG.train_save_path['npy'], train_features)
        train_idxs = mine_nearest_neighbors(train_features, topk=20)
        np.save(CFG.train_save_path['knn'], train_idxs)
    
    if CFG.extract_test_feats:
        print('starting extract testing set features')
        T = ValT(CFG.img_size)
        test_dataset =  BaseData(transform=T, train=False, download=False)
        test_data_loader = DataLoader(test_dataset, shuffle=False, batch_size=16, drop_last=False)
        print(f'len test data {len(test_dataset)}')
        test_features = get_all_features(test_data_loader, model, CFG.device)
        print('test features shape:', test_features.shape)
        np.save(CFG.test_save_path['npy'], test_features)
        test_idxs = mine_nearest_neighbors(test_features, topk=20)
        np.save(CFG.test_save_path['knn'], test_idxs)

if __name__ == '__main__':
    main()