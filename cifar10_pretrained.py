
import os
import sys
import torch

from cifar10_paths import models_path, state_dict_path

sys.path.append(models_path)

import cifar10_models

def get_model(config, pretrained=False):
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
