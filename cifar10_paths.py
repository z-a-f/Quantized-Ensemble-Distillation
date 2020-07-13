import os

##############
# Loacal paths
data_path = os.path.expanduser(os.path.join('~', 'data', 'CIFAR'))
models_path = os.path.expanduser(os.path.join('~', 'Git', 'PyTorch_CIFAR10'))
state_dict_path = os.path.join(models_path, 'downloads', 'cifar10_models', 'state_dicts')
student_save_path = '.'
