from __future__ import print_function
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from vision.torchvision.datasets.smallnorb import smallNORB

from model import capsules, CNN
from loss import SpreadLoss

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Matrix-Capsules-EM')
parser.add_argument('--model', type=str, default='em-capsules', metavar='M',
                    help='Neural network model')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 8)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 8)')
parser.add_argument('--test-intvl', type=int, default=1, metavar='N',
                    help='test intvl (default: 1)')
parser.add_argument('--test-size', type=float, default=.01, metavar='N',
                    help='percentage of the test set used for calculating accuracy (default: 1%%)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--weight-decay', type=float, default=0, metavar='WD',
                    help='weight decay (default: 0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--em-iters', type=int, default=3, metavar='N',
                    help='iterations of EM Routing (default: 3)')
parser.add_argument('--snapshot-folder', type=str, default='./snapshots', metavar='SF',
                    help='where to store the snapshots')
parser.add_argument('--data-folder', type=str, default='./data', metavar='DF',
                    help='where to store the datasets')
parser.add_argument('--dataset', type=str, default='smallNORB', metavar='D',
                    help='dataset for training(mnist, smallNORB)')

def get_setting(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    path = os.path.join(args.data_folder, args.dataset)
    if args.dataset == 'mnist':
        num_class = 10
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'smallNORB':
        num_class = 5
        train_loader = torch.utils.data.DataLoader(
            smallNORB(path, train=True, download=True,
                      transform=transforms.Compose([
                          transforms.Resize(48),
                          transforms.RandomCrop(32),
                          transforms.ColorJitter(brightness=32./255, contrast=0.5),
                          transforms.ToTensor()#,
                          #transforms.Normalize((0, 0, 0), (1, 1, 1))
                      ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            smallNORB(path, train=False,
                      transform=transforms.Compose([
                          transforms.Resize(48),
                          transforms.CenterCrop(32),
                          transforms.ToTensor()
                      ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        raise NameError('Undefined dataset {}'.format(args.dataset))
    return num_class, train_loader, test_loader

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

def train(train_loader, model, criterion, optimizer, epoch, device):
    batch_time = AverageMeter()

    model.train()
    train_len = len(train_loader)
    epoch_acc = 0
    toc = time.time()
    running_acc = 0
    running_loss = 0

    for batch_idx, (data, target, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        r = (1.*batch_idx + (epoch-1)*train_len) / (args.epochs*train_len)
        loss = criterion(output, target, r)
        acc = accuracy(output, target)
        loss.backward()
        optimizer.step()
        
        epoch_acc += acc
        running_acc += acc
        running_loss += loss.item()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {}\t[{}/{} ({:.0f}%)]\t'
                  'Loss: {:.3f}\tAccuracy: {:.1f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader),
                  running_loss/args.log_interval, running_acc/args.log_interval,
                  batch_time=batch_time))
            running_acc = 0
            running_loss = 0
            
            batch_time.update(time.time() - toc)
            toc = time.time()
    return epoch_acc

def snapshot(model, folder, epoch):
    path = os.path.join(folder, 'model_{}.pth'.format(epoch))
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    print('saving model to {}'.format(path))
    torch.save(model.state_dict(), path)

def test(test_loader, model, criterion, device, chunk=.01):
    model.eval()
    test_loss = 0
    acc = 0
    test_len = len(test_loader)
    tested = 0
    with torch.no_grad():
        for data, target, _ in test_loader:
            if tested/(args.batch_size*test_len) >= chunk:
                break
            data, target = data.to(device), target.to(device)
            output = model(data)
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

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")
    
    # datasets
    num_class, train_loader, test_loader = get_setting(args)

    # model
    if args.model == 'em-capsules':
        A, B, C, D = 64, 8, 16, 16
        # A, B, C, D = 32, 32, 32, 32
        model = capsules(A=A, B=B, C=C, D=D, E=num_class,
                         iters=args.em_iters, device=device)
    elif args.model == 'cnn':
        model = CNN(num_class)
        model.to(device)
    print('Training %d parameters' % (count_parameters(model)))

    criterion = SpreadLoss(num_class=num_class, m_min=0.2, m_max=0.9, device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1)

    best_acc = test(test_loader, model, criterion, device, chunk=args.test_size)
    try:
        for epoch in range(1, args.epochs + 1):
            acc = train(train_loader, model, criterion, optimizer, epoch, device)
            acc /= len(train_loader)
            scheduler.step(acc)
            if epoch % args.test_intvl == 0:
                best_acc = max(best_acc, test(test_loader, model, criterion, device, chunk=args.test_size))
    except KeyboardInterrupt:
        print('cancelled training after %d epochs' % (epoch - 1))
    best_acc = max(best_acc, test(test_loader, model, criterion, device, chunk=args.test_size))
    print('best test accuracy: {:.6f}'.format(best_acc))

    snapshot(model, args.snapshot_folder, args.epochs)

if __name__ == '__main__':
    main()
