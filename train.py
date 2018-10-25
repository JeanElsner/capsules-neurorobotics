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

    model, criterion, optimizer, scheduler = load_model(
        args.model, device_ids=args.device_ids, lr=args.lr, routing_iters=args.routing_iters)

    # datasets
    num_class, train_loader, test_loader = load_datasets(
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
    result = os.path.join('.', 'results', 'pytorch_train.csv')
    append_to_csv(result, to_write, header=header)


if __name__ == '__main__':
    main()
