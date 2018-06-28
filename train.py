from __future__ import print_function
import argparse
import os
import time
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from train import get_smallNORB_test_data, get_smallNORB_train_data
from model import MatrixCapsules, CNN, VectorCapsules
from loss import SpreadLoss
from utils import AverageMeter, accuracy, count_parameters, snapshot

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Matrix-Capsules-EM')
parser.add_argument('--model', type=str, default='matrix-capsules', metavar='M',
                    help='Neural network model')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 56)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 56)')
parser.add_argument('--test-intvl', type=int, default=1, metavar='N',
                    help='test intvl (default: 1)')
parser.add_argument('--test-size', type=float, default=.1, metavar='N',
                    help='percentage of the test set used for calculating accuracy (default: 0.05)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.08)')
parser.add_argument('--weight-decay', type=float, default=0, metavar='WD',
                    help='weight decay (default: 0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
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
parser.add_argument('--start-epoch', type=int, default=1, metavar='N',
                    help='Epoch to start training at')

def get_setting(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    path = os.path.join(args.data_dir, args.dataset)
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
        train_loader = get_smallNORB_train_data(path, args.batch_size, cuda=args.cuda, shuffle=True)
        test_loader = get_smallNORB_test_data(path, args.test_batch_size, cuda=args.cuda, shuffle=True)
    else:
        raise NameError('Undefined dataset {}'.format(args.dataset))
    return num_class, train_loader, test_loader

def train(train_loader, model, criterion, optimizer, epoch, device):
    batch_time = AverageMeter()

    model.train()
    train_len = len(train_loader)
    epoch_acc = 0
    toc = time.time()
    running_acc = 0
    running_loss = 0

    for batch_idx, (data, target, _) in enumerate(train_loader):
        r = (1.*batch_idx + (epoch-1)*train_len) / (args.epochs*train_len)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, r)
        loss = criterion(output, target, r)
        acc = accuracy(output, target)
        loss.backward()
        optimizer.step()

        epoch_acc += acc
        running_acc += acc
        running_loss += loss.item()

        if batch_idx % args.log_interval == args.log_interval - 1:
            batch_time.update(time.time() - toc)
            toc = time.time()
            print('Train Epoch: {}\t[{}/{} ({:.0f}%)]\t'
                  'Loss: {:.3f}\tAccuracy: {:.1f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                  epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader),
                  running_loss/args.log_interval, running_acc/args.log_interval,
                  batch_time=batch_time))
            running_acc = 0
            running_loss = 0
    return epoch_acc/len(train_loader)

def test(test_loader, model, criterion, device, chunk=.01):
    return 0
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
            output = model(data, 1)
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
    num_class, train_loader, test_loader = get_setting(args)

    # model
    if args.model == 'matrix-capsules':
        A, B, C, D = 64, 8, 16, 16
        #A, B, C, D = 32, 32, 32, 32
        model = MatrixCapsules(A=A, B=B, C=C, D=D, E=num_class, 
                               iters=args.em_iters, device=device,
                               _lambda=[[1e-4, 1e-2], [1e-4, 1e-2], [1e-4, 1e-2]])
    elif args.model == 'cnn':
        model = CNN(num_class)
    elif args.model == 'vector-capsules':
        model = VectorCapsules(3, num_classes=num_class)
    model.to(device)
    print('Training %d parameters' % (count_parameters(model)))

    if args.snapshot:
        print('Pretrained model from snapshot %s' % (args.snapshot))
        model.load_state_dict(torch.load(args.snapshot))

    criterion = SpreadLoss(num_class=num_class, m_min=0.2, m_max=0.9, device=device)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)#, momentum=.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1)

    best_acc = test(test_loader, model, criterion, device, chunk=args.test_size)
    try:
        for epoch in range(args.start_epoch, args.epochs + 1):
            print()
            acc = train(train_loader, model, criterion, optimizer, epoch, device)
            scheduler.step(acc)
            print('Epoch accuracy was %.1f%%. Learning rate is %.9f.' % 
                  (acc, optimizer.state_dict()['param_groups'][0]['lr']))
            if epoch % args.test_intvl == 0:
                best_acc = max(best_acc, test(test_loader, model, criterion, device, chunk=args.test_size))
    except KeyboardInterrupt:
        print('Cancelled training after %d epochs' % (epoch - 1))
        args.epochs = epoch - 1

    best_acc = max(best_acc, test(test_loader, model, criterion, device, chunk=args.test_size))
    print('Best test accuracy: {:.6f}'.format(best_acc))

    snapshot(model, args.snapshot_dir, args.epochs)

if __name__ == '__main__':
    main()
