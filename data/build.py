import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, io, models, ops, transforms, utils

from data.datasets import CIFAR10Dataset, MediumImagenetHDF5Dataset


def build_loader(config):
    if config.DATA.DATASET == "cifar10":
        dataset_train = CIFAR10Dataset(img_size=config.DATA.IMG_SIZE, train=True)
        dataset_val = CIFAR10Dataset(img_size=config.DATA.IMG_SIZE, train=False)
        dataset_test = CIFAR10Dataset(img_size=config.DATA.IMG_SIZE, train=False)
    elif config.DATA.DATASET == "medium_imagenet":
        dataset_train = MediumImagenetHDF5Dataset(config.DATA.IMG_SIZE, split="train", augment=True)
        dataset_val = MediumImagenetHDF5Dataset(config.DATA.IMG_SIZE, split="val", augment=False)
        dataset_test = MediumImagenetHDF5Dataset(config.DATA.IMG_SIZE, split="test", augment=False)
    else:
        raise NotImplementedError

    ######## Added by Anthony
    data_sample = DataLoader(
        dataset_train,
        batch_size=10,
        shuffle=True,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
    )
    #########

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
    )

    image, _ = next(iter(data_loader_train))

    image = torch.nn.functional.normalize(image[0], p=2.0, dim=1).permute(1, 2, 0)
    plt.imshow(image)
    plt.plot()
    plt.show()

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=False,
    )

    data_loader_test = DataLoader(
        dataset_test,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=False,
    )

    return dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test
