"""
transforms for dataset
"""

from torchvision import transforms
from src_py.augs import Augment, Cutout


class Base:
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]


class ContrastiveTransformations(object):
    """
    return 2 augmented image for one single image
    """
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views
        
    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]


class SimclrT:
    """
    transform for training simclr
    """
    def __init__(self, img_size):
        mean = Base.mean
        std = Base.std
        self.T = transforms.Compose([
                transforms.RandomResizedCrop(size=(img_size, img_size), scale=[0.2, 1.0]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
                ], p=0.8),
                transforms.RandomGrayscale(0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            
    def __call__(self, x):
        return self.T(x)


class ScanT:
    """
    transforms for training scan and self label
    """
    def __init__(self, img_size, num_strong_aug=4 , cut_out_size=16):
        # this is the transforms you must need as discribed in the paper
        mean = Base.mean
        std = Base.std
        self.T = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            Augment(num_strong_aug),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            Cutout(1, cut_out_size, True)
            ])
            
    def __call__(self, x):
        return self.T(x)


class ValT:
    """
    transform for validation
    """
    def __init__(self, img_size):
        mean = Base.mean
        std = Base.std
        self.T = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        
    def __call__(self, x):
        return self.T(x)
