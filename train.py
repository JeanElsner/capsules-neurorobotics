import argparse
import os
import numpy as np
import torch
from datasets import VPRTorch
from model import MatrixCapsules, CNN, VectorCapsules
from loss import SpreadLoss
from utils import count_parameters, snapshot, append_to_csv, path_to_save_string
from torchvision import transforms
from training import test, train

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
parser.add_argument('--dataset', type=str, default=r'./data/Dataset_lighting3/left', metavar='DD',
                    help='path to the dataset')
parser.add_argument('--inv-temp', type=float, default=1e-3, metavar='N',
                    help='Inverse temperature parameter for the EM algorithm')
parser.add_argument('--device-ids', nargs='+', default=[0], type=int)
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--append', default=False, action='store_true')
parser.add_argument('--split-angle', default=False, action='store_true')
parser.add_argument('--azimuth', default=list(range(1, 19)), nargs='+', type=int)
parser.add_argument('--elevation', default=list(range(0, 9)), nargs='+', type=int)


def get_vpr_train_data(path, batch_size, azimuth, elevation, cuda=True):
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    vpr = VPRTorch(path, train=True,
                   transform=transforms.Compose([
                       #                       transforms.Resize(48),
                       #                       transforms.RandomCrop(32),
                       #                       transforms.ColorJitter(brightness=32. / 255, contrast=0.5),
                       transforms.ToTensor()  # ,
                       # transforms.Normalize((0, 0, 0), (1, 1, 1))
                   ]),
                   azimuth=azimuth, elevation=elevation)
    train_loader = torch.utils.data.DataLoader(
        vpr, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader


def get_vpr_test_data(path, batch_size, azimuth, elevation, cuda=True):
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    vpr = VPRTorch(path, train=False,
                   transform=transforms.Compose([
                       #                      transforms.Resize(48),
                       #                      transforms.CenterCrop(32),
                       transforms.ToTensor()
                   ]),
                   azimuth=azimuth, elevation=elevation)
    test_loader = torch.utils.data.DataLoader(
        vpr, batch_size=batch_size, shuffle=False, **kwargs)
    return test_loader


def get_setting(path, batch_size, test_batch_size, cuda, split_angle, azimuth, elevation):
    num_class = 5
    train_loader = get_vpr_train_data(path, batch_size, azimuth, elevation, cuda=cuda)

    if split_angle:
        test_azimuth = np.arange(1, 19)[[i not in azimuth for i in range(1, 19)]]
        test_elevation = np.arange(0, 9)[[i not in elevation for i in range(0, 9)]]
    else:
        test_azimuth, test_elevation = (azimuth, elevation)
    test_loader = get_vpr_test_data(path, test_batch_size, test_azimuth, test_elevation, cuda=cuda)
    print(f'{len(train_loader.dataset)} training images, {len(test_loader.dataset)} test images')
    return num_class, train_loader, test_loader


def main():
    global args
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print()
    print('Command-line argument values:')
    for key, value in vars(args).items():
        print('-', key, ':', value)
    print()

    params = [
        args.model, path_to_save_string(args.dataset), args.split_angle, args.azimuth, args.elevation, args.batch_size,
        args.epochs, args.lr, args.weight_decay, args.seed, args.em_iters
    ]
    model_name = '_'.join([str(x) for x in params]) + '.pth'
    header = 'model,dataset,split_angle,azimuth,elevation,batch_size,epochs,lr,weight_decay,seed,em_iters,accuracy'
    path = os.path.join('.', 'snapshots', model_name)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    # datasets
    num_class, train_loader, test_loader = get_setting(
        args.dataset, args.batch_size, args.test_batch_size, args.cuda, args.split_angle, args.azimuth, args.elevation)

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
    print(f'Network has {count_parameters(model):d} parameters')

    criterion = SpreadLoss(num_class=num_class, m_min=0.2, m_max=0.9, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # , momentum=.8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1)

    best_acc = 0

    if args.test or args.append:
        model.load_state_dict(torch.load(path))
    if not args.test:
        try:
            for epoch in range(1, args.epochs + 1):
                print()
                acc = train(train_loader, model, criterion, optimizer, epoch, device, epochs=args.epochs, log_interval=args.log_interval)
                scheduler.step(acc)
                print('Epoch accuracy was %.1f%%. Learning rate is %.9f.' %
                      (acc, optimizer.state_dict()['param_groups'][0]['lr']))
                if epoch % args.test_interval == 0:
                    test_acc = test(test_loader, model, criterion, device, chunk=args.test_size)
                    if test_acc > best_acc:
                        best_acc = test_acc
        except KeyboardInterrupt:
            print('Cancelled training after %d epochs' % (epoch - 1))
            args.epochs = epoch - 1
        snapshot(path, model)
    acc = test(test_loader, model, criterion, device, chunk=1)
    print(f'Accuracy: {acc:.2f}% (best: {best_acc:.2f}%)')

    to_write = params + [acc.cpu().numpy()]
    result = os.path.join('.', 'results', 'pytorch_test.csv' if args.test else 'pytorch_train.csv')
    append_to_csv(result, to_write, header=header)


if __name__ == '__main__':
    main()
