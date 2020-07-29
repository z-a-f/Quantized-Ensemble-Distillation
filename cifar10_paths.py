import os
import sys

##############
# Loacal paths
data_path = os.path.expanduser(os.path.join('~', 'data', 'CIFAR'))
models_path = os.path.abspath(os.path.join('.', 'PyTorch_CIFAR10'))
state_dict_path = os.path.join(models_path, 'cifar10_models', 'state_dicts')

if not os.path.isdir(state_dict_path):
  raise RuntimeError('Please run `git submodule --init --recursive` and run `cifar10_download.py` from the `PyTorch_CIFAR10` directory.')

sys.path.append(models_path)
