import argparse
import os
import numpy as np
import torch
from model import load_model
from datasets import load_datasets
from utils import snapshot, append_to_csv, path_to_save_string, add_training_arguments, make_dirs_if_not_exist
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
    snapshot_path = os.path.join('.', 'snapshots', model_name)
    data_path = os.path.join('.', 'results', 'training_data', model_name)
    result_path = os.path.join('.', 'results', 'pytorch_train.csv')

    make_dirs_if_not_exist([snapshot_path, data_path, result_path])

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model, criterion, optimizer, scheduler = load_model(
        args.model, device_ids=args.device_ids, lr=args.lr, routing_iters=args.routing_iters)
    num_class, train_loader, test_loader = load_datasets(
        args.dataset, args.batch_size, args.test_batch_size, args.viewpoint_modulo)

    best_acc = 0
    training_accuracies = []
    test_accuracies = []

    if args.append:
        model.load_state_dict(torch.load(snapshot_path))
    try:
        for epoch in range(1, args.epochs + 1):
            print()
            acc = train(train_loader, model, criterion, optimizer, epoch, epochs=args.epochs, log_interval=args.log_interval)
            training_accuracies.append(acc)
            scheduler.step(acc)
            print('Epoch accuracy was %.1f%%. Learning rate is %.9f.' %
                  (acc, optimizer.state_dict()['param_groups'][0]['lr']))
            if epoch % args.test_interval == 0:
                test_acc, __, __, __ = test(test_loader, model, criterion, chunk=args.test_size)
                test_accuracies.append(test_acc)
                if test_acc > best_acc:
                    best_acc = test_acc
    except KeyboardInterrupt:
        print('Cancelled training after %d epochs' % (epoch - 1))
        args.epochs = epoch - 1

    acc, predictions, labels, logits = test(test_loader, model, criterion, chunk=1)
    print(f'Accuracy: {acc:.2f}% (best: {best_acc:.2f}%)')

    to_write = params + [acc.cpu().numpy()]
    append_to_csv(result_path, to_write, header=header)
    snapshot(snapshot_path, model)
    #torch.save((accuracies, labels, predictions), data_path)

    if args.learn_curve != '':
        make_dirs_if_not_exist(args.learn_curve)
        torch.save((training_accuracies, test_accuracies), args.learn_curve)


if __name__ == '__main__':
    main()
