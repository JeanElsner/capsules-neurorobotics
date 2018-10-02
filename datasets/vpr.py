# -*- coding: utf-8 -*-

import glob
import torch
import os
import numpy as np
import cv2
import math
import re
from bindsnet.datasets import Dataset
from typing import Tuple, List, Iterable, Any, Dict


class VPR(Dataset):
    """ Visual Pattern Recognition Dataset.
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

    def __init__(self, path: str) -> None:
        if not os.path.isdir(path):
            raise FileNotFoundError('Path doesn\'t exist.')
        self.path = path
        # super().__init__(path, download)

    def get_train(
            self,
            elevation: Tuple[int, int] = (0, 8),
            azimuth: Tuple[int, int] = (1, 18)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not os.path.isfile(os.path.join(self.path, VPR.train_images_pickle)) \
                or not os.path.isfile(os.path.join(self.path, VPR.train_labels_pickle)) \
                or not os.path.isfile(os.path.join(self.path, VPR.train_orientation_pickle)):
            files = os.path.join(self.path, f'*.{VPR.ext}')
            files = glob.glob(files)
            if len(files) < 1:
                raise FileNotFoundError('No images found in path.')
            images, labels, orientation = self.process_images(files)
            torch.save(images, open(os.path.join(self.path, VPR.train_images_pickle), 'wb'))
            torch.save(labels, open(os.path.join(self.path, VPR.train_labels_pickle), 'wb'))
            torch.save(orientation, open(os.path.join(self.path, VPR.train_orientation_pickle), 'wb'))
        else:
            print('Loading training images from serialized object file.\n')
            images = torch.load(open(os.path.join(self.path, VPR.train_images_pickle), 'rb'))
            labels = torch.load(open(os.path.join(self.path, VPR.train_labels_pickle), 'rb'))
            orientation = torch.load(open(os.path.join(self.path, VPR.train_orientation_pickle), 'rb'))

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

        return red_images, red_labels

    def get_test(
            self,
            elevation: Tuple[int, int] = (0, 8),
            azimuth: Tuple[int, int] = (1, 18)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not os.path.isfile(os.path.join(self.path, VPR.test_images_pickle)) \
                or not os.path.isfile(os.path.join(self.path, VPR.test_labels_pickle)) \
                or not os.path.isfile(os.path.join(self.path, VPR.test_orientation_pickle)):
            files = os.path.join(self.path, f'*.{VPR.ext}')
            files = glob.glob(files)
            if len(files) < 1:
                raise FileNotFoundError('No images found in path.')
            images, labels, orientation = self.process_images(files, train=False)
            torch.save(images, open(os.path.join(self.path, VPR.test_images_pickle), 'wb'))
            torch.save(labels, open(os.path.join(self.path, VPR.test_labels_pickle), 'wb'))
            torch.save(orientation, open(os.path.join(self.path, VPR.test_orientation_pickle), 'wb'))
        else:
            print('Loading training images from serialized object file.\n')
            images = torch.load(open(os.path.join(self.path, VPR.test_images_pickle), 'rb'))
            labels = torch.load(open(os.path.join(self.path, VPR.test_labels_pickle), 'rb'))
            orientation = torch.load(open(os.path.join(self.path, VPR.test_orientation_pickle), 'rb'))

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

        return red_images, red_labels

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
            result = re.search(f'(.+)(\d+)_(\d+)_(\d+)\\.{VPR.ext}$', fn)
            if train and int(result.group(2)) not in VPR.training_instances:
                continue
            elif not train and int(result.group(2)) not in VPR.test_instances:
                continue
            labels = np.append(labels, VPR.label_dict[result.group(1)])
            orientation = np.append(orientation, (int(result.group(3)), int(result.group(4))))
            im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            im = cv2.resize(im, VPR.im_size)
            images = np.append(images, im)
        images = images.reshape((-1, *VPR.im_size))
        orientation = orientation.reshape((images.shape[0], -1))
        return images, labels, orientation
