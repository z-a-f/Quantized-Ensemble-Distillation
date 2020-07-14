
import importlib
import os
import sys
import torch

from cifar10_paths import models_path, state_dict_path

def get_model(config, pretrained=False):
  sys.path.append(models_path)
  import cifar10_models

  model_name = config['classifier']
  model_class = cifar10_models
  for name in model_name.split('.'):
    model_class = getattr(model_class, name)
  model = model_class(pretrained=False, progress=True)

  if pretrained:
    state_dict_name = config['state_dict']
    state_dict = torch.load(os.path.join(state_dict_path, state_dict_name))
    model.load_state_dict(state_dict)

  return model, model_name

def get_qat_model(config, pretrained=False):
  name_to_import = 'qat_models'
  model_name = config['classifier']
  qat_folder = importlib.import_module(name_to_import)
  model_class = getattr(qat_folder, model_name)
  model = model_class(pretrained=False, progress=True)

  if pretrained:
    state_dict_name = config['state_dict']
    state_dict = torch.load(os.path.join(state_dict_path, state_dict_name))
    model.load_state_dict(state_dict)

  return model, model_name + '_qat'

