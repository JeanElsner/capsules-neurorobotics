import torch
from torchvision import datasets, transforms
from datasets import VPRTorch
import numpy as np

from torch.utils.data.sampler import Sampler


class SolidAngleSampler(Sampler):
    def __init__(self, sn):
        if sn.train:
            meta = sn.train_orientation
            indices = np.where((meta[:, 0] < 7) & (meta[:, 0] > 1) & (meta[:, 1] < 32) & (meta[:, 1] > 2))[0]
        else:
            meta = sn.test_orientation
            indices = np.where((meta[:, 0] >= 7) | (meta[:, 0] <= 1) | (meta[:, 1] >= 32) | (meta[:, 1] <= 2))[0]
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class OneShotSampler(Sampler):
    def __init__(self, sn):
        if sn.train:
            meta = sn.train_orientation
            indices = np.where(meta[:, 0] == 9)[0]
        else:
            meta = sn.test_orientation
            indices = range(0, len(meta))
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class StrideSampler(Sampler):
    def __init__(self, stride, sn):
        if sn.train:
            meta = sn.train_info
            indices = range(0, len(meta), stride)
        else:
            meta = sn.test_info
            indices = range(1, len(meta), stride)
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def get_VPR_train_data(path, batch_size, cuda=True, shuffle=True):
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    vpr = VPRTorch(path, train=True, download=True,
                   transform=transforms.Compose([
#                       transforms.Resize(48),
#                       transforms.RandomCrop(32),
#                       transforms.ColorJitter(brightness=32. / 255, contrast=0.5),
                       transforms.ToTensor()  # ,
                       # transforms.Normalize((0, 0, 0), (1, 1, 1))
                   ]))
    train_loader = torch.utils.data.DataLoader(
        vpr, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader


def get_VPR_test_data(path, batch_size, cuda=True, shuffle=True):
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    vpr = VPRTorch(path, train=False, download=True,
                   transform=transforms.Compose([
 #                      transforms.Resize(48),
 #                      transforms.CenterCrop(32),
                       transforms.ToTensor()
                   ]))
    test_loader = torch.utils.data.DataLoader(
        vpr, batch_size=batch_size, shuffle=False, **kwargs)
    return test_loader


if __name__ == '__main__':
    vpr = VPRTorch('../data/VPR', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize(48),
                       transforms.CenterCrop(32)
                   ]))
    # instance, elevation, azimuth, lighting
    # classes = ['animal', 'human', 'plane', 'truck', 'car']
    elevations = [30, 35, 40, 45, 50, 55, 60, 65, 70]
    # azimuth*10
    if vpr.train:
        meta = vpr.train_info
        meta = np.where((meta[:, 1] < 7) & (meta[:, 1] > 1) & (meta[:, 2] < 32) & (meta[:, 2] > 2))
        print(meta[0])
    else:
        meta = vpr.test_info
        meta = meta[np.where(((meta[:, 1] >= 7) | (meta[:, 1] <= 1)) & ((meta[:, 2] >= 32) | (meta[:, 2] <= 2)))]
    print((meta[i] for i in torch.randperm(len(meta))))
