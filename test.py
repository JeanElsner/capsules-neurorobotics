import argparse
import os
import numpy as np
import torch
from model import load_model
from datasets import load_datasets
from utils import snapshot, append_to_csv, path_to_save_string, add_training_arguments
from training import test, train

# Training arguments
parser = argparse.ArgumentParser(description='Visual Pattern Recognition')
add_training_arguments(parser)
parser.add_argument('--test-dataset', type=str, default=r'./data/Dataset_lighting1/left')


def main():
    global args
    args = parser.parse_args()

    print()
    print('Command-line argument values:')
    for key, value in vars(args).items():
        print('-', key, ':', value)
    print()

    model_params = [
        args.model, path_to_save_string(args.dataset), args.viewpoint_modulo, args.batch_size,
        args.epochs, args.lr, args.weight_decay, args.seed, args.routing_iters
    ]
    model_name = '_'.join([str(x) for x in model_params]) + '.pth'
    header = 'model,training-dataset,test-dataset,viewpoint_modulo,' \
             'batch_size,epochs,lr,weight_decay,seed,em_iters,accuracy'
    path = os.path.join('.', 'snapshots', model_name)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model, criterion, optimizer, scheduler = load_model(
        args.model, device_ids=args.device_ids, lr=args.lr, routing_iters=args.routing_iters)

    # datasets
    num_class, train_loader, test_loader = load_datasets(
        args.test_dataset, args.batch_size, args.test_batch_size, args.viewpoint_modulo)
    model.load_state_dict(torch.load(path))
    acc = test(test_loader, model, criterion, chunk=1)
    print(f'Accuracy: {acc:.2f}%')

    test_params = [
        args.model, path_to_save_string(args.dataset), path_to_save_string(args.test_dataset),
        args.viewpoint_modulo, args.batch_size, args.epochs, args.lr, args.weight_decay, args.seed, args.routing_iters
    ]
    to_write = test_params + [acc.cpu().numpy()]
    result = os.path.join('.', 'results', 'pytorch_test.csv')
    append_to_csv(result, to_write, header=header)


if __name__ == '__main__':
    main()
