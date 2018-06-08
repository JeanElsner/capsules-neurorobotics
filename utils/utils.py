import os
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def accuracy(output, target):
    return (target==output.max(1)[1]).sum().float()*100/len(target)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def snapshot(model, folder, epoch):
    path = os.path.join(folder, '{}_{}.pth'.format(model.__class__.__name__, epoch))
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    print('Saving model to {}'.format(path))
    torch.save(model.state_dict(), path)