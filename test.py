import argparse
import os
import numpy as np
import torch
from datasets import VPRTorch
from model import load_model
from utils import snapshot, append_to_csv, path_to_save_string
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
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many mini-batches to wait before logging training status')
parser.add_argument('--routing-iters', type=int, default=2, metavar='N',
                    help='iterations of dynamic routing')
parser.add_argument('--dataset', type=str, default=r'./data/Dataset_lighting3/left', metavar='DD',
                    help='path to the dataset')
parser.add_argument('--inv-temp', type=float, default=1e-3, metavar='N',
                    help='Inverse temperature parameter for the EM algorithm')
parser.add_argument('--device-ids', nargs='+', default=[0], type=int)
parser.add_argument('--viewpoint-modulo', type=int, default=1)


def get_vpr_train_data(path, batch_size, azimuth, elevation):
    kwargs = {'num_workers': 1, 'pin_memory': True}
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


def get_vpr_test_data(path, batch_size, azimuth, elevation):
    kwargs = {'num_workers': 1, 'pin_memory': True}
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


def get_settings(path, batch_size, test_batch_size, viewpoint_modulo):
    num_class = 5
    azimuth = np.arange(1, 19, viewpoint_modulo)
    elevation = np.arange(0, 9, viewpoint_modulo)
    train_loader = get_vpr_train_data(path, batch_size, azimuth, elevation)
    test_loader = get_vpr_test_data(path, test_batch_size, np.arange(1, 19), np.arange(0, 9))
    print(f'{len(train_loader.dataset)} training images, {len(test_loader.dataset)} test images')
    return num_class, train_loader, test_loader


def main():
    global args
    args = parser.parse_args()

    print()
    print('Command-line argument values:')
    for key, value in vars(args).items():
        print('-', key, ':', value)
    print()

    params = [
        args.model, path_to_save_string(args.dataset), args.viewpoint_modulo, args.batch_size,
        args.epochs, args.lr, args.weight_decay, args.seed, args.routing_iters
    ]
    model_name = '_'.join([str(x) for x in params]) + '.pth'
    header = 'model,dataset,viewpoint_modulo,batch_size,epochs,lr,weight_decay,seed,em_iters,accuracy'
    path = os.path.join('.', 'snapshots', model_name)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model, criterion, optimizer, scheduler = load_model(args.model, device_ids=args.device_ids, lr=args.lr, weight_decay=args.weight_decay, routing_iters=args.routing_iters)

    # datasets
    num_class, train_loader, test_loader = get_settings(
        args.dataset, args.batch_size, args.test_batch_size, args.viewpoint_modulo)

    best_acc = 0

    if args.append:
        model.load_state_dict(torch.load(path))

    try:
        for epoch in range(1, args.epochs + 1):
            print()
            acc = train(train_loader, model, criterion, optimizer, epoch, epochs=args.epochs, log_interval=args.log_interval)
            scheduler.step(acc)
            print('Epoch accuracy was %.1f%%. Learning rate is %.9f.' %
                  (acc, optimizer.state_dict()['param_groups'][0]['lr']))
            if epoch % args.test_interval == 0:
                test_acc = test(test_loader, model, criterion, chunk=args.test_size)
                if test_acc > best_acc:
                    best_acc = test_acc
    except KeyboardInterrupt:
        print('Cancelled training after %d epochs' % (epoch - 1))
        args.epochs = epoch - 1
    snapshot(path, model)
    acc = test(test_loader, model, criterion, chunk=1)
    print(f'Accuracy: {acc:.2f}% (best: {best_acc:.2f}%)')

    to_write = params + [acc.cpu().numpy()]
    result = os.path.join('.', 'results', 'pytorch_test.csv')
    append_to_csv(result, to_write, header=header)


if __name__ == '__main__':
    main()
