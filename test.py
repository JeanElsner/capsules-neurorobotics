from __future__ import print_function
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from train import get_smallNORB_test_data, get_smallNORB_train_data
from model import MatrixCapsules, CNN, VectorCapsules
from loss import SpreadLoss

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Matrix-Capsules-EM')
parser.add_argument('--model', type=str, default='matrix-capsules', metavar='M',
                    help='Neural network model')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 56)')
parser.add_argument('--test-size', type=float, default=1, metavar='N',
                    help='percentage of the test set used for calculating accuracy (default: 0.05)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--em-iters', type=int, default=2, metavar='N',
                    help='iterations of EM Routing (default: 3)')
parser.add_argument('--snapshot-dir', type=str, default='./snapshots', metavar='SD',
                    help='where to store the snapshots')
parser.add_argument('--data-dir', type=str, default='./data', metavar='DD',
                    help='where to store the datasets')
parser.add_argument('--dataset', type=str, default='smallNORB', metavar='D',
                    help='dataset for training(mnist, smallNORB)')
parser.add_argument('--inv-temp', type=float, default=1e-3, metavar='N',
                    help='Inverse temperature parameter for the EM algorithm')
parser.add_argument('--snapshot', type=str, default='', metavar='SNP',
                    help='Use pretrained network from snapshot')

def get_setting(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    path = os.path.join(args.data_dir, args.dataset)
    if args.dataset == 'mnist':
        num_class = 10
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'smallNORB':
        num_class = 5
        test_loader = get_smallNORB_test_data(path, args.test_batch_size, cuda=args.cuda, shuffle=True)
    else:
        raise NameError('Undefined dataset {}'.format(args.dataset))
    return num_class, test_loader

def accuracy(output, target):
    return (target==output.max(1)[1]).sum().float()*100/len(target)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

def test(test_loader, model, criterion, device, chunk=.01):
    model.eval()
    test_loss = 0
    acc = 0
    test_len = len(test_loader)
    tested = 0
    with torch.no_grad():
        for data, target, _ in test_loader:
            if tested/(args.test_batch_size*test_len) >= chunk:
                break
            data, target = data.to(device), target.to(device)
            output = model(data, 12/100)
            test_loss += criterion(output, target, r=1).item()
            acc += accuracy(output, target)
            tested += len(data)

    test_loss /= test_len
    acc /= (test_len*chunk)
    print('\nTest set: Average loss: {:.6f}, Accuracy: {:.6f} \n'.format(
        test_loss, acc))
    return acc

def main():
    global args, best_prec1
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    #torch.manual_seed(args.seed)
    #if args.cuda:
    #    torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")
    
    # datasets
    num_class, test_loader = get_setting(args)

    # model
    if args.model == 'matrix-capsules':
        A, B, C, D = 64, 8, 16, 16
        model = MatrixCapsules(A=A, B=B, C=C, D=D, E=num_class, 
                               iters=args.em_iters, device=device,
                               _lambda=[[1e-4, 1e-2], [1e-4, 1e-2], [1e-4, 1e-2]])
    elif args.model == 'cnn':
        model = CNN(num_class)
        model.to(device)
    elif args.model == 'vector-capsules':
        model = VectorCapsules(3, num_classes=num_class)
        model.to(device)
    print('%d trained parameters' % (count_parameters(model)))

    if args.snapshot:
        print('Pretrained model from snapshot %s' % (args.snapshot))
        model.load_state_dict(torch.load(args.snapshot))

    criterion = SpreadLoss(num_class=num_class, m_min=0.2, m_max=0.9, device=device)
    acc = test(test_loader, model, criterion, device, chunk=args.test_size)
    print('Test set accuracy: {:.6f}'.format(acc))

if __name__ == '__main__':
    main()
