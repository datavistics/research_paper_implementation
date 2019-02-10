# Special thanks to Amir Saniyan: https://github.com/amir-saniyan/AlexNet/blob/master/dataset_helper.py
# The CIFAR-10 dataset:
# https://www.cs.toronto.edu/~kriz/cifar.html

import torch
from src.data_processing.utilities import download_and_unzip
from global_fun import *
import pickle
import numpy as np
import scipy.misc
import shutil

batch_1_file = cifar_10_dir / 'data_batch_1'
batch_2_file = cifar_10_dir / 'data_batch_2'
batch_3_file = cifar_10_dir / 'data_batch_3'
batch_4_file = cifar_10_dir / 'data_batch_4'
batch_5_file = cifar_10_dir / 'data_batch_5'
test_batch_file = cifar_10_dir / 'test_batch'

module_logger = module_logging(__file__, False)


@logspeed(module_logger)
def get_cifar_10(force_download: bool=False):
    """
    ``get_cifar_10`` will:
        - Download data
        - Unzip data
        - Place data in correct dir

    :param force_download: Will delete current download and re-download
    :type force_download: bool
    """
    if force_download:
        module_logger.info("Forcing Download")
        if cifar_10_dir.exists():
            shutil.rmtree(cifar_10_dir)

    tar_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    outfile_path = data_dir

    if batch_1_file.exists():
        module_logger.info("Data Already Exists...")
    else:
        module_logger.info("Getting Cifar 10 Data...")
        download_and_unzip(tar_url, outfile_path)


def __unpickle(file):
    with open(file, 'rb') as fo:
        unpickled_obj = pickle.load(fo, encoding='bytes')
    return unpickled_obj


@logspeed(module_logger)
def read_cifar_10(image_width: int, image_height: int, device: torch.device):
    """
    Reads data and returns train/test split.

    :param image_width:
    :type image_width: int
    :param image_height:
    :type image_height: int
    :return: X_train, y_train, X_test, y_test
    :rtype: tuple(np.array)
    """
    batch_1 = __unpickle(batch_1_file)
    batch_2 = __unpickle(batch_2_file)
    batch_3 = __unpickle(batch_3_file)
    batch_4 = __unpickle(batch_4_file)
    batch_5 = __unpickle(batch_5_file)
    test_batch = __unpickle(test_batch_file)

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    total_train_samples = len(batch_1[b'labels']) + len(batch_2[b'labels']) + len(batch_3[b'labels']) \
                          + len(batch_4[b'labels']) + len(batch_5[b'labels'])

    X_train = np.zeros(shape=[total_train_samples, image_width, image_height, 3], dtype=np.uint8)
    Y_train = np.zeros(shape=[total_train_samples, len(classes)], dtype=np.float32)

    batches = [batch_1, batch_2, batch_3, batch_4, batch_5]

    index = 0
    for batch in batches:
        for i in range(len(batch[b'labels'])):
            image = batch[b'data'][i].reshape(3, 32, 32).transpose([1, 2, 0])
            label = batch[b'labels'][i]

            X = scipy.misc.imresize(image, size=(image_height, image_width), interp='bicubic')
            Y = np.zeros(shape=[len(classes)], dtype=np.int)
            Y[label] = 1

            X_train[index + i] = X
            Y_train[index + i] = Y

        index += len(batch[b'labels'])

    total_test_samples = len(test_batch[b'labels'])

    X_test = np.zeros(shape=[total_test_samples, image_width, image_height, 3], dtype=np.uint8)
    Y_test = np.zeros(shape=[total_test_samples, len(classes)], dtype=np.float32)

    for i in range(len(test_batch[b'labels'])):
        image = test_batch[b'data'][i].reshape(3, 32, 32).transpose([1, 2, 0])
        label = test_batch[b'labels'][i]

        X = scipy.misc.imresize(image, size=(image_height, image_width), interp='bicubic')
        Y = np.zeros(shape=[len(classes)], dtype=np.int)
        Y[label] = 1

        X_test[i] = torch.tensor(X, device=device)
        Y_test[i] = torch.tensor(Y, device=device)

    return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':
    get_cifar_10()
