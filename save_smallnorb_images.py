from __future__ import print_function
import argparse
import os
from train import get_smallNORB_test_data, get_smallNORB_train_data
import torchvision

parser = argparse.ArgumentParser(description='Saves the processed smallNORB images')
parser.add_argument('--data-folder', type=str, default='./data', metavar='DF',
                    help='Path to where the smallNORB dataset is stored')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

classes = ['animal', 'human', 'plane', 'truck', 'car']
elevations = [30, 35, 40, 45, 50, 55, 60, 65, 70]

def main(_args):
    path = os.path.join(_args.data_folder, 'smallNORB')
    args = [path, 1]
    kwargs = {'cuda':not _args.no_cuda, 'shuffle':False}
    save_path= os.path.join(path, 'images')
    
    if not os.path.exists(os.path.join(save_path, 'train')):
        os.makedirs(os.path.join(save_path, 'train'))

    for image, label, meta in get_smallNORB_train_data(*args, **kwargs):
        meta = meta.squeeze()
        name = '%s_%d_%d_%d_%d.jpg' % (classes[label], meta[0], elevations[meta[1]], meta[2]*10, meta[3])
        torchvision.utils.save_image(image, os.path.join(save_path, 'train', name))
    
    if not os.path.exists(os.path.join(save_path, 'test')):
        os.makedirs(os.path.join(save_path, 'test'))
    
    for image, label, meta in get_smallNORB_test_data(*args, **kwargs):
        meta = meta.squeeze()
        name = '%s_%d_%d_%d_%d.jpg' % (classes[label], meta[0], elevations[meta[1]], meta[2]*10, meta[3])
        torchvision.utils.save_image(image, os.path.join(save_path, 'test', name))

if __name__ == '__main__':
    main(parser.parse_args())