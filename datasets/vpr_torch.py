from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import struct
import math
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch
import torch.utils.data as data
import random
from typing import Dict, Tuple, List
import glob
import cv2
import re

class VPRTorch(data.Dataset):
    """ Visual Pattern Recognition Dataset for torch.
    Args:
        root (string): Root directory of dataset
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    train_images_pickle: str = 'train_images.pt'
    train_orientation_pickle: str = 'train_orientation.pt'
    train_labels_pickle: str = 'train_labels.pt'
    test_images_pickle: str = 'test_images.pt'
    test_orientation_pickle: str = 'test_orientation.pt'
    test_labels_pickle: str = 'test_labels.pt'

    ext: str = 'png'
    im_size: Tuple[int, int] = (32, 32)
    training_instances = [1, 2, 3, 4, 5]
    test_instances = [6, 7, 8]

    label_dict: Dict = {'chair': 0, 'cutlery': 1, 'lighter': 2, 'chess': 3, 'cup': 4}
    
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, seed = 0):
        self.seed = seed
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train: bool = train  # training set or test set

        if self.train:
            self.train_data, self.train_labels, self.train_orientation = self.get_train()
        else:
            self.test_data, self.test_labels, self.test_orientation = self.get_test()

    def get_train(
            self,
            elevation: Tuple[int, int] = (0, 8),
            azimuth: Tuple[int, int] = (1, 18)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not os.path.isfile(os.path.join(self.root, VPRTorch.train_images_pickle)) \
                or not os.path.isfile(os.path.join(self.root, VPRTorch.train_labels_pickle)) \
                or not os.path.isfile(os.path.join(self.root, VPRTorch.train_orientation_pickle)):
            files = os.path.join(self.root, f'*.{VPRTorch.ext}')
            files = glob.glob(files)
            if len(files) < 1:
                raise FileNotFoundError('No images found in path.')
            images, labels, orientation = self.process_images(files)
            torch.save(images, open(os.path.join(self.root, VPRTorch.train_images_pickle), 'wb'))
            torch.save(labels, open(os.path.join(self.root, VPRTorch.train_labels_pickle), 'wb'))
            torch.save(orientation, open(os.path.join(self.root, VPRTorch.train_orientation_pickle), 'wb'))
        else:
            print('Loading training images from serialized object file.\n')
            images = torch.load(open(os.path.join(self.root, VPRTorch.train_images_pickle), 'rb'))
            labels = torch.load(open(os.path.join(self.root, VPRTorch.train_labels_pickle), 'rb'))
            orientation = torch.load(open(os.path.join(self.root, VPRTorch.train_orientation_pickle), 'rb'))

        indices = np.all([
            orientation[:, 0] >= elevation[0],
            orientation[:, 0] <= elevation[1],
            orientation[:, 1] >= azimuth[0],
            orientation[:, 1] <= azimuth[1]], axis=0)
        red_images = torch.Tensor(images[indices])
        red_labels = torch.Tensor(labels[indices])
        red_orientation = torch.Tensor(orientation[indices])
        permute = np.random.permutation(red_labels.shape[0])
        red_images = red_images[permute]
        red_labels = red_labels[permute]
        red_orientation = red_orientation[permute]

        return red_images, red_labels, red_orientation

    def get_test(
            self,
            elevation: Tuple[int, int] = (0, 8),
            azimuth: Tuple[int, int] = (1, 18)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not os.path.isfile(os.path.join(self.root, VPRTorch.test_images_pickle)) \
                or not os.path.isfile(os.path.join(self.root, VPRTorch.test_labels_pickle)) \
                or not os.path.isfile(os.path.join(self.root, VPRTorch.test_orientation_pickle)):
            files = os.path.join(self.root, f'*.{VPRTorch.ext}')
            files = glob.glob(files)
            if len(files) < 1:
                raise FileNotFoundError('No images found in path.')
            images, labels, orientation = self.process_images(files, train=False)
            torch.save(images, open(os.path.join(self.root, VPRTorch.test_images_pickle), 'wb'))
            torch.save(labels, open(os.path.join(self.root, VPRTorch.test_labels_pickle), 'wb'))
            torch.save(orientation, open(os.path.join(self.root, VPRTorch.test_orientation_pickle), 'wb'))
        else:
            print('Loading test images from serialized object file.\n')
            images = torch.load(open(os.path.join(self.root, VPRTorch.test_images_pickle), 'rb'))
            labels = torch.load(open(os.path.join(self.root, VPRTorch.test_labels_pickle), 'rb'))
            orientation = torch.load(open(os.path.join(self.root, VPRTorch.test_orientation_pickle), 'rb'))

        indices = np.all([
            orientation[:, 0] >= elevation[0],
            orientation[:, 0] <= elevation[1],
            orientation[:, 1] >= azimuth[0],
            orientation[:, 1] <= azimuth[1]], axis=0)
        red_images = torch.Tensor(images[indices])
        red_labels = torch.Tensor(labels[indices])
        red_orientation = torch.Tensor(orientation[indices])
        permute = np.random.permutation(red_labels.shape[0])
        red_images = red_images[permute]
        red_labels = red_labels[permute]
        red_orientation = red_orientation[permute]

        return red_images, red_labels, red_orientation

    @staticmethod
    def process_images(files: List, train: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        files_l = len(files)
        images = np.array([], dtype=np.uint8)
        labels = np.array([], dtype=np.uint8)
        orientation = np.array([], dtype=np.uint8)
        idx = 0
        width = math.ceil(math.log10(files_l))
        tenth = files_l * .1
        for f in files:
            idx += 1
            if files_l % idx == tenth:
                print(f'Preprocessing images {idx:{width}}/{files_l:{width}}')
            __, fn = os.path.split(f)
            result = re.search(f'(.+)(\d+)_(\d+)_(\d+)\\.{VPRTorch.ext}$', fn)
            if train and int(result.group(2)) not in VPRTorch.training_instances:
                continue
            elif not train and int(result.group(2)) not in VPRTorch.test_instances:
                continue
            labels = np.append(labels, VPRTorch.label_dict[result.group(1)])
            orientation = np.append(orientation, (int(result.group(3)), int(result.group(4))))
            im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            im = cv2.resize(im, VPRTorch.im_size)
            images = np.append(images, im)
        images = images.reshape((-1, *VPRTorch.im_size))
        orientation = orientation.reshape((images.shape[0], -1))
        return images, labels, orientation

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        #dindex = math.floor(index/2)
        if self.train:
            img, target, orientation = self.train_data[index], self.train_labels[index], self.train_orientation[index]
        else:
            img, target, orientation = self.test_data[index], self.test_labels[index], self.test_orientation[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy())
        target = target.long()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return self.train_labels.shape[0]
        else:
            return self.test_labels.shape[0]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
