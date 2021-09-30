import os
import argparse
import pickle
import numpy as np
from PIL import Image


def save_png(data_dir, data_name, save_dir):
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    d = unpickle(os.path.join(data_dir, data_name))
    images = d[b'data'].reshape([-1, 3, 32, 32]).transpose(0, 2, 3, 1)
    labels = d[b'labels']
    filenames = [f.decode() for f in d[b'filenames']]
    path_list = [f'{save_dir}/{name} {label}' for name, label in zip(filenames, labels)]
    for image, name in zip(images, filenames):
        with Image.fromarray(image) as img:
            img.save(f'{save_dir}/{name}')
    return path_list


def save_path(path_list, save_path):
    string = '\n'.join(path_list)
    with open(save_path, 'wt') as f:
        f.write(string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    args = parser.parse_args()

    # training data
    save_dir = os.path.abspath('./cifar10/train')
    os.makedirs(save_dir, exist_ok=True)
    path_list = []
    for i in range(1, 6):
        path_list += save_png(args.data_dir, f'data_batch_{i}', save_dir)
    save_path(path_list, save_dir + '.txt')

    # test data
    save_dir = os.path.abspath('./cifar10/test')
    os.makedirs(save_dir, exist_ok=True)
    path_list = save_png(args.data_dir, 'test_batch', save_dir)
    save_path(path_list, save_dir + '.txt')
