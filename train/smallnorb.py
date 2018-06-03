import torch
from torchvision import datasets, transforms
from vision.torchvision.datasets.smallnorb import smallNORB

def get_smallNORB_train_data(path, batch_size, cuda=True, shuffle=True):
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        smallNORB(path, train=True, download=True,
                  transform=transforms.Compose([
                      transforms.Resize(48),
                      transforms.RandomCrop(32),
                      transforms.ColorJitter(brightness=32./255, contrast=0.5),
                      transforms.ToTensor()#,
                      #transforms.Normalize((0, 0, 0), (1, 1, 1))
                  ])),
        batch_size=batch_size, shuffle=shuffle, **kwargs)
    return train_loader
    
def get_smallNORB_test_data(path, batch_size, cuda=True, shuffle=True):
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    test_loader = torch.utils.data.DataLoader(
        smallNORB(path, train=False,
                  transform=transforms.Compose([
                      transforms.Resize(48),
                      transforms.CenterCrop(32),
                      transforms.ToTensor()
                  ])),
        batch_size=batch_size, shuffle=shuffle, **kwargs)
    return test_loader