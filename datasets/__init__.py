from .vpr import VPR
from .vpr_torch import VPRTorch
import torch
import numpy as np
from torchvision.transforms import transforms


def load_datasets(path, batch_size, test_batch_size, viewpoint_modulo):
    num_class = 5
    azimuth = np.arange(1, 19, viewpoint_modulo)
    elevation = np.arange(0, 9, viewpoint_modulo)
    train_loader = get_vpr_data_loader(path, batch_size, azimuth, elevation)
    test_loader = get_vpr_data_loader(path, test_batch_size, np.arange(1, 19), np.arange(0, 9), train=False)
    print(f'{len(train_loader.dataset)} training images, {len(test_loader.dataset)} test images')
    return num_class, train_loader, test_loader


def get_vpr_data_loader(path, batch_size, azimuth, elevation, train=True):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    vpr = VPRTorch(path, train=train,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ]),
                   azimuth=azimuth, elevation=elevation)
    data_loader = torch.utils.data.DataLoader(
        vpr, batch_size=batch_size, shuffle=False, **kwargs)
    return data_loader
