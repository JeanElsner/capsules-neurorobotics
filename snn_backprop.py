import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from time import time as t
from sklearn.metrics import confusion_matrix

from bindsnet.datasets import MNIST
from bindsnet.network import Network, load_network
from bindsnet.utils import get_square_weights
from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Connection
from bindsnet.network.nodes import RealInput, IFNodes, DiehlAndCookNodes
from bindsnet.analysis.plotting import plot_spikes, plot_weights

from datasets import VPR
from bindsnet.learning import PostPre

# Parameters.
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_hidden', type=int, default=800)
parser.add_argument('--n_train', type=int, default=3000)
parser.add_argument('--n_test', type=int, default=1000)
parser.add_argument('--time', default=15, type=int)
parser.add_argument('--lr', default=0.01, type=float)
#parser.add_argument('--lr', default=0.006, type=float)
#parser.add_argument('--lr', default=0.00001, type=float)
parser.add_argument('--lr_decay', default=.5, type=float)
parser.add_argument('--decay_memory', default=5, type=int)
parser.add_argument('--update_interval', default=100, type=int)
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--plot', dest='plot', action='store_true')
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--test', dest='train', action='store_false')
parser.add_argument('--gpu', dest='gpu', action='store_true')
parser.set_defaults(plot=False, train=False, gpu=False)
args = parser.parse_args()

seed = args.seed  # random seed
n_hidden = args.n_hidden  # no. of hidden layer neurons
n_train = args.n_train  # no. of training samples
n_test = args.n_test  # no. of test samples
time = args.time  # simulation time
lr = args.lr  # learning rate
lr_decay = args.lr_decay  # learning rate decay
update_interval = args.update_interval  # no. examples between evaluation
plot = args.plot  # visualize spikes + connection weights
train = args.train  # train or test mode
gpu = args.gpu  # whether to use gpu or cpu tensors
epochs = args.epochs
decay_memory = args.decay_memory

args = vars(args)

print()
print('Command-line argument values:')
for key, value in args.items():
    print('-', key, ':', value)
print()

data = 'vpr'
model = 'two_layer_backprop'

assert n_train % update_interval == 0 and n_test % update_interval == 0, \
                        'No. examples must be divisible by update_interval'

params = [
    seed, n_hidden, n_train, time, lr, lr_decay, update_interval
]

model_name = '_'.join([str(x) for x in params])

if not train:
    test_params = [
        seed, n_hidden, n_train, n_test, time, lr, lr_decay, update_interval
    ]

np.random.seed(seed)

if gpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)

# Paths.
top_level = os.path.join('.', 'results')
data_path = os.path.join('.', 'data', 'Dataset_lighting4', 'left')
params_path = os.path.join(top_level, 'params', data, model)
curves_path = os.path.join(top_level, 'curves', data, model)
results_path = os.path.join(top_level, 'results', data, model)
confusion_path = os.path.join(top_level, 'confusion', data, model)

for path in [params_path, curves_path, results_path, confusion_path]:
    if not os.path.isdir(path):
        os.makedirs(path)

criterion = torch.nn.CrossEntropyLoss()  # Loss function on output firing rates.
sqrt = int(np.ceil(np.sqrt(n_hidden)))  # Ceiling(square root(no. hidden neurons)).
n_examples = n_train if train else n_test

if train:
    # Network building.
    network = Network()

    # Groups of neurons.
    input_layer = RealInput(n=32**2, sum_input=True)
    hidden_layer = IFNodes(n=n_hidden, sum_input=True, traces=True)
    hidden_bias = RealInput(n=1, sum_input=True)
    output_layer = IFNodes(n=5, sum_input=True)
    output_bias = RealInput(n=1, sum_input=True)
    network.add_layer(input_layer, name='X')
    network.add_layer(hidden_layer, name='Y')
    network.add_layer(hidden_bias, name='Y_b')
    network.add_layer(output_layer, name='Z')
    network.add_layer(output_bias, name='Z_b')

    recurrent_connection = Connection(source=hidden_layer, target=hidden_layer, update_rule=PostPre,
                                #norm=78.4,
                                nu_pre=1e-4,
                                nu_post=1e-2,
                                wmax=1.0, wmin=-1)
    # Connections between groups of neurons.
    input_connection = Connection(source=input_layer, target=hidden_layer)
    hidden_bias_connection = Connection(source=hidden_bias, target=hidden_layer)
    hidden_connection = Connection(source=hidden_layer, target=output_layer)
    output_bias_connection = Connection(source=output_bias, target=output_layer)
    network.add_connection(input_connection, source='X', target='Y')
    network.add_connection(hidden_bias_connection, source='Y_b', target='Y')
    network.add_connection(hidden_connection, source='Y', target='Z')
    network.add_connection(output_bias_connection, source='Z_b', target='Z')
    #network.add_connection(recurrent_connection, source='Y', target='Y')

    # State variable monitoring.
    for l in network.layers:
        m = Monitor(network.layers[l], state_vars=['s'], time=time)
        network.add_monitor(m, name=l)
else:
    network = load_network(os.path.join(params_path, model_name + '.pt'))

# Load MNIST data.
dataset = VPR(data_path)

if train:
    _images, _labels = dataset.get_train()
else:
    epochs = 1
    _images, _labels = dataset.get_test()

n_examples = _images.shape[0]

# Run training.
start = beginning = t()
mean_acc = []
mean_best = -np.inf
last_improv = 0
for epoch in range(epochs):
    images, labels = _images[:n_examples], _labels[:n_examples]
    images, labels = iter(images.view(-1, 32 ** 2) / 255), iter(labels)
    grads = {}
    accuracies = []
    predictions = []
    ground_truth = []
    best = -np.inf
    spike_ims, spike_axes, weights1_im, weights2_im = None, None, None, None
    losses = torch.zeros(update_interval)
    correct = torch.zeros(update_interval)
    for i, (image, label) in enumerate(zip(images, labels)):
        label = torch.Tensor([label]).long()

        # Run simulation for single datum.
        inpts = {
            'X': image.repeat(time, 1), 'Y_b': torch.ones(time, 1), 'Z_b': torch.ones(time, 1)
        }
        network.run(inpts=inpts, time=time)

        # Retrieve spikes and summed inputs from both layers.
        spikes = {l: network.monitors[l].get('s') for l in network.layers if not '_b' in l}
        summed_inputs = {l: network.layers[l].summed / time for l in network.layers}

        # Compute softmax of output spiking activity and get predicted label.
        output = summed_inputs['Z'].softmax(0).view(1, -1)
        predicted = output.argmax(1).item()
        correct[i % update_interval] = int(predicted == label[0].item())
        predictions.append(predicted)
        ground_truth.append(label)

        # Compute cross-entropy loss between output and true label.
        losses[i % update_interval] = criterion(output, label)

        if train:
            # Compute gradient of the loss WRT average firing rates.
            grads['dl/df2'] = summed_inputs['Z'].softmax(0)
            grads['dl/df2'][label] -= 1

            # Compute gradient of the summed voltages WRT connection weights.
            # This is an approximation; the summed voltages are not a
            # smooth function of the connection weights.
            grads['dl/dw2'] = torch.ger(summed_inputs['Y'], grads['dl/df2'])
            grads['dl/db2'] = grads['dl/df2']
            grads['dl/dw1'] = torch.ger(summed_inputs['X'], network.connections['Y', 'Z'].w @ grads['dl/df2'])
            grads['dl/db1'] = network.connections['Y', 'Z'].w @ grads['dl/df2']
            #grads['dl/df1'] =

            # Do stochastic gradient descent calculation.
            network.connections['X', 'Y'].w -= lr * grads['dl/dw1']
            network.connections['Y_b', 'Y'].w -= lr * grads['dl/db1']
            network.connections['Y', 'Z'].w -= lr * grads['dl/dw2']
            network.connections['Z_b', 'Z'].w -= lr * grads['dl/db2']

        if i > 0 and i % update_interval == 0:
            accuracies.append(correct.mean() * 100)
            mean_acc.append(np.mean(accuracies))

            if train:
                if len(mean_acc) >= decay_memory:
                    mean_acc = mean_acc[-decay_memory:]
                    if torch.Tensor([i >= mean_best for i in mean_acc]).sum() < 1:
                        if last_improv < 1:
                            last_improv = decay_memory
                            print()
                            print(f'No improvements in {decay_memory} intervals, decaying learning rate.')
                            lr *= lr_decay
                        else:
                            last_improv -= 1
                if mean_acc[-1] > mean_best:
                    mean_best = mean_acc[-1]
                if accuracies[-1] > best:
                    print()
                    print('New best accuracy! Saving network parameters to disk.')

                    # Save network to disk.
                    network.save(os.path.join(params_path, model_name + '.pt'))
                    best = accuracies[-1]

            print()
            print(f'Epoch {epoch+1} of {epochs}')
            print(f'Progress: {i} / {n_examples} ({t() - start:.3f} seconds)')
            print(f'Average cross-entropy loss: {losses.mean():.3f}')
            print(f'Last interval accuracy: {accuracies[-1]:.3f}')
            print(f'Average accuracy: {mean_acc[-1]:.3f}')

            if train:
                print(f'Best average accuracy: {mean_best:.4f}')
                print(f'Current learning rate: {lr:.5f}')

            start = t()

        if plot:
            w = network.connections['Y', 'Z'].w
            weights = [
                w[:, i].view(sqrt, sqrt) for i in range(5)
            ]
            w = torch.zeros(5*sqrt, 2*sqrt)
            for i in range(5):
                for j in range(2):
                    w[i*sqrt: (i+1)*sqrt, j*sqrt: (j+1)*sqrt] = weights[i + j * 5]

            spike_ims, spike_axes = plot_spikes(spikes, ims=spike_ims, axes=spike_axes)
            weights1_im = plot_weights(w, im=weights1_im, wmin=-1, wmax=1)

            w = network.connections['X', 'Y'].w
            square_weights = get_square_weights(w, sqrt, 28)
            weights2_im = plot_weights(square_weights, im=weights2_im, wmin=-1, wmax=1)

            plt.pause(1e-8)

        network.reset_()  # Reset state variables.

accuracies.append(correct.mean() * 100)

if train:
    lr *= lr_decay

    if accuracies[-1] > best:
        print()
        print('New best accuracy! Saving network parameters to disk.')

        # Save network to disk.
        network.save(os.path.join(params_path, model_name + '.pt'))
        best = accuracies[-1]

print()
print(f'Progress: {n_examples} / {n_examples} ({t() - start:.3f} seconds)')
print(f'Average cross-entropy loss: {losses.mean():.3f}')
print(f'Last accuracy: {accuracies[-1]:.3f}')
print(f'Average accuracy: {np.mean(accuracies):.3f}')

if train:
    print(f'Best accuracy: {best:.3f}')

if train:
    print('\nTraining complete.\n')
else:
    print('\nTest complete.\n')

print(f'Average accuracy: {np.mean(accuracies):.3f}')

# Save accuracy curves to disk.
to_write = ['train'] + params if train else ['test'] + params
f = '_'.join([str(x) for x in to_write]) + '.pt'
torch.save((accuracies, update_interval, n_examples), open(os.path.join(curves_path, f), 'wb'))

results = [np.mean(accuracies), np.max(accuracies)]
to_write = params + results if train else test_params + results
to_write = [str(x) for x in to_write]
name = 'train.csv' if train else 'test.csv'

if not os.path.isfile(os.path.join(results_path, name)):
    with open(os.path.join(results_path, name), 'w') as f:
        if train:
            f.write(
                'seed,n_hidden,n_train,time,lr,lr_decay,update_interval,mean_accuracy,max_accuracy\n'
            )
        else:
            f.write(
                'seed,n_hidden,n_train,n_test,time,lr,lr_decay,update_interval,mean_accuracy,max_accuracy\n'
            )

with open(os.path.join(results_path, name), 'a') as f:
    f.write(','.join(to_write) + '\n')

# Compute confusion matrices and save them to disk.
confusion = confusion_matrix(ground_truth, predictions)

to_write = ['train'] + params if train else ['test'] + test_params
f = '_'.join([str(x) for x in to_write]) + '.pt'
torch.save(confusion, os.path.join(confusion_path, f))

print()
