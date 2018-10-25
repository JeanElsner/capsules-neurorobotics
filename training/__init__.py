import torch
import time
from utils import AverageMeter, accuracy_from_final_layer


def train(train_loader, model, criterion, optimizer, epoch, epochs=10, log_interval=1):
    batch_time = AverageMeter()

    model.train()
    train_len = len(train_loader)
    epoch_acc = 0
    toc = time.time()
    running_acc = 0
    running_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        r = (1. * batch_idx + (epoch - 1) * train_len) / (epochs * train_len)
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data, r)
        loss = criterion(output, target, r)
        acc = accuracy_from_final_layer(output, target)
        loss.backward()
        optimizer.step()

        epoch_acc += acc
        running_acc += acc
        running_loss += loss.item()

        if batch_idx % log_interval == log_interval - 1:
            batch_time.update(time.time() - toc)
            print('Train Epoch: {}\t[{}/{} ({:.0f}%)]\t'
                  'Loss: {:.3f}\tAccuracy: {:.1f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(epoch, (batch_idx + 1) * len(data),
                                                                              len(train_loader.dataset),
                                                                              100. * batch_idx / len(train_loader),
                                                                              running_loss / log_interval,
                                                                              running_acc / log_interval,
                                                                              batch_time=batch_time))
            running_acc = 0
            running_loss = 0
    return epoch_acc / len(train_loader)


def test(test_loader, model, criterion, chunk=.01):
    model.eval()
    test_loss = 0
    acc = 0
    test_len = len(test_loader)
    tested = 0
    with torch.no_grad():
        for data, target in test_loader:
            if tested / (test_loader.batch_size * test_len) >= chunk:
                break
            data, target = data.cuda(), target.cuda()
            output = model(data, 1)
            test_loss += criterion(output, target, r=1).item()
            acc += accuracy_from_final_layer(output, target)
            tested += len(data)
    test_loss /= (test_len * chunk)
    acc /= (test_len * chunk)
    print(f'Test set: Average loss: {test_loss:.6f}, Accuracy: {acc:.6f}')
    print()
    return acc
