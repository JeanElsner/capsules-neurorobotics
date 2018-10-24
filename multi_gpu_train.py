import argparse
import os
import numpy as np
import time
import torch
import torch.optim as optim
from datasets import get_VPR_test_data, get_VPR_train_data
from model import MatrixCapsules, CNN, VectorCapsules
from loss import SpreadLoss
from utils import AverageMeter, accuracy, count_parameters, snapshot

# Training settings
parser = argparse.ArgumentParser(description='Visual Pattern Recognition')
parser.add_argument('--model', type=str, default='vector-capsules', metavar='M',
                    help='Neural network model')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing')
parser.add_argument('--test-interval', type=int, default=1, metavar='N',
                    help='interval in epochs to test')
parser.add_argument('--test-size', type=float, default=1, metavar='N',
                    help='percentage of the test set used for calculating accuracy during training')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate')
parser.add_argument('--weight-decay', type=float, default=0, metavar='WD',
                    help='weight decay')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many mini-batches to wait before logging training status')
parser.add_argument('--em-iters', type=int, default=2, metavar='N',
                    help='iterations of EM Routing')
parser.add_argument('--dataset', type=str, default=r'./data/Dataset_lighting4/left', metavar='DD',
                    help='path to the dataset')
parser.add_argument('--inv-temp', type=float, default=1e-3, metavar='N',
                    help='Inverse temperature parameter for the EM algorithm')
parser.add_argument('--device-ids', nargs='+', default=[0], type=int)


def get_setting(path, batch_size, test_batch_size, cuda):
    num_class = 5
    train_loader = get_VPR_train_data(path, batch_size, cuda=cuda, shuffle=True)
    test_loader = get_VPR_test_data(path, test_batch_size, cuda=cuda, shuffle=True)
    return num_class, train_loader, test_loader


def train(train_loader, model, criterion, optimizer, epoch, device):
    batch_time = AverageMeter()

    model.train()
    train_len = len(train_loader)
    epoch_acc = 0
    toc = time.time()
    running_acc = 0
    running_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        r = (1. * batch_idx + (epoch - 1) * train_len) / (args.epochs * train_len)
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
                       running_loss / args.log_interval, running_acc / args.log_interval,
                batch_time=batch_time))
            running_acc = 0
            running_loss = 0
    return epoch_acc / len(train_loader)


def test(test_loader, model, criterion, device, chunk=.01):
    model.eval()
    test_loss = 0
    acc = 0
    test_len = len(test_loader)
    tested = 0
    with torch.no_grad():
        for data, target in test_loader:
            if tested / (args.test_batch_size * test_len) >= chunk:
                break
            data, target = data.to(device), target.to(device)
            output = model(data, 1)
            test_loss += criterion(output, target, r=1).item()
            acc += accuracy(output, target)
            tested += len(data)

    test_loss /= test_len
    acc /= (test_len * chunk)
    print()
    print(f'Test set: Average loss: {test_loss:.6f}, Accuracy: {acc:.6f}')
    print()
    return acc


def main():
    global args
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print()
    print('Command-line argument values:')
    for key, value in vars(args).items():
        print('-', key, ':', value)
    print()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    # datasets
    num_class, train_loader, test_loader = get_setting(args.dataset, args.batch_size, args.test_batch_size, args.cuda)

    # model
    if args.model == 'matrix-capsules':
        A, B, C, D = 64, 8, 16, 16
        # A, B, C, D = 32, 32, 32, 32
        model = MatrixCapsules(A=A, B=B, C=C, D=D, E=num_class,
                               iters=args.em_iters, device=device,
                               _lambda=[[1e-4, 1e-2], [1e-4, 1e-2], [1e-4, 1e-2]])
    elif args.model == 'cnn':
        model = CNN(num_class)
    elif args.model == 'vector-capsules':
        model = VectorCapsules(3, num_classes=num_class)
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=args.device_ids)
    print(f'Training {count_parameters(model):d} parameters')

    params = [
        args.model, args.batch_size, args.epochs, args.lr, args.weight_decay, args.seed, args.em_iters
    ]
    model_name = '_'.join([str(x) for x in params]) + '.pth'
    path = os.path.join('.', 'snapshots', model_name)

    criterion = SpreadLoss(num_class=num_class, m_min=0.2, m_max=0.9, device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # , momentum=.8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1)

    best_acc = test(test_loader, model, criterion, device, chunk=args.test_size)
    try:
        for epoch in range(1, args.epochs + 1):
            print()
            acc = train(train_loader, model, criterion, optimizer, epoch, device)
            scheduler.step(acc)
            print('Epoch accuracy was %.1f%%. Learning rate is %.9f.' %
                  (acc, optimizer.state_dict()['param_groups'][0]['lr']))
            if epoch % args.test_interval == 0:
                test_acc = test(test_loader, model, criterion, device, chunk=args.test_size)
                if test_acc > best_acc:
                    best_acc = test_acc
                    snapshot(path, model)
    except KeyboardInterrupt:
        print('Cancelled training after %d epochs' % (epoch - 1))
        args.epochs = epoch - 1

    best_acc = max(best_acc, test(test_loader, model, criterion, device, chunk=args.test_size))
    print('Best test accuracy: {:.6f}'.format(best_acc))

    to_write = params + [best_acc.cpu().numpy()]
    to_write = [str(x) for x in to_write]
    result = os.path.join('.', 'results', 'pytorch_train.csv')

    if not os.path.exists(os.path.dirname(result)):
        os.makedirs(os.path.dirname(result))

    if not os.path.isfile(result):
        with open(result, 'w') as f:
            f.write(
                'mode,batch_size,epochs,lr,weight_decay,seed,em_iters,max_accuracy\n'
            )
    with open(result, 'a') as f:
        f.write(','.join(to_write) + '\n')


if __name__ == '__main__':
    main()
