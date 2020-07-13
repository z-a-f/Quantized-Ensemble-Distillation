from collections.abc import Iterable
import os

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as T

from cifar10_paths import data_path

kMean = [0.4914, 0.4822, 0.4465]
kStd = [0.2023, 0.1994, 0.2010]

def get_train_loader(batch_size=512):
  transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(kMean, kStd)
  ])
  train_set = CIFAR10(data_path, train=True, transform=transform, download=True)
  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                            num_workers=os.cpu_count(), pin_memory=True,
                            drop_last=False)
  return train_loader

def get_test_loader(batch_size=512):
  transform = T.Compose([
    T.ToTensor(),
    T.Normalize(kMean, kStd)
  ])
  test_set = CIFAR10(data_path, train=False, transform=transform, download=True)
  test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                           num_workers=os.cpu_count(), pin_memory=True,
                           drop_last=False)
  return test_loader
