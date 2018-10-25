import os
import torch
import re


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


def accuracy_from_final_layer(output, target):
    return (target == output.max(1)[1]).sum().float() * 100 / len(target)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def snapshot(path, model):
    make_dirs_if_not_exist(path)
    print(f'Saving model to {path}')
    torch.save(model.state_dict(), path)


def make_dirs_if_not_exist(dirs):
    if not isinstance(dirs, (list, tuple)):
        dirs = [dirs]
    for d in dirs:
        if not os.path.exists(os.path.dirname(d)):
            os.makedirs(os.path.dirname(d))


def path_to_save_string(path, replace='_'):
    return re.sub('[\\/]', replace, os.path.dirname(path))


def append_to_csv(path, to_write, header=''):
    to_write = [str(x) for x in to_write]
    make_dirs_if_not_exist(path)
    if not os.path.isfile(path):
        with open(path, 'w') as f:
            f.write(header + '\n')
    with open(path, 'a') as f:
        f.write(','.join(to_write) + '\n')
