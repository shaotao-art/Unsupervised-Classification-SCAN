"""
dataset for training simclr, scan, self-label
"""

from torch.utils.data import Dataset
import numpy as np
import torchvision


class BaseData(Dataset):
    """
    dataset to load image
    """
    def __init__(self, transform=None, train=True, download=True):
        super(BaseData, self).__init__()
        self.datas = torchvision.datasets.CIFAR10(root='/content/data', train=train, download=download)
        self.T = transform

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        # only return images
        # do not return labels
        if self.T is None:
            return self.datas[index][0]
        else:
            return self.T(self.datas[index][0])



class SimclrData(Dataset):
    """
    dataset to train simclr step
    this dataset need contrastive transform: return two augmented image for one single image
    """
    def __init__(self, dataset, transform):
        super(SimclrData, self).__init__()
        # get data from BaseData
        self.datas = dataset
        self.image_transform = transform

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        out = self.image_transform(self.datas[index])
        return out


class ScanData(Dataset):
    """
    dataset for scan step
    this dataset need strong Augment as discribed in paper
    """
    def __init__(self, dataset, indices, transform):
        super(ScanData, self).__init__()
        self.data = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        out = dict()
        out['anchor'] = self.transform(self.data[idx])
        nei_idx = np.random.choice(self.indices[idx], 1)[0]
        out['neighbor'] = self.transform(self.data[nei_idx])
        return out

    def __len__(self):
        return self.indices.shape[0]


class SelfLabelData(Dataset):
    """
    dataset for selflabel step
    return one weak augmented image and one strong augmented image
    """
    def __init__(self, dataset, weak_T, strong_T):
        super(SelfLabelData, self).__init__()
        self.datas = dataset
        self.weak_T = weak_T
        self.strong_T = strong_T

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        out = dict()
        image = self.datas[idx]
        out['image'] = self.weak_T(image)
        out['image_augmented'] = self.strong_T(image)
        return out

