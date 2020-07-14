import argparse
import os
from math import ceil
import sys
import yaml
import json

import numpy as np
import torch

from cifar10_data import get_train_loader, get_test_loader
from cifar10_loss import soft_logloss, num_correct as num_correct_fn
from cifar10_optimizer import get_optimizer
from cifar10_paths import models_path, state_dict_path
from cifar10_pretrained import get_model
from cifar10_run import epoch_self_distillation_train
from cifar10_run import epoch_self_distillation_test

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('teacher_config_file', type=str, help='YAML configuration file for the teacher')
  parser.add_argument('student_config_file', type=str, help='YAML configuration file for the student')
  parser.add_argument('--preload_student', action='store_const', const=True, default=False)
  parser.add_argument('--seed', type=int, default=None, help='Repro seed')
  parser.add_argument('--device', type=str, default=None, help='Force device')
  parser.add_argument('--batch_size', type=int, default=None)
  parser.add_argument('--save_to', type=str, default='results')

  return parser.parse_args()

def seed_all(seed):
  if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_loaders(batch_size):
  train_loader = get_train_loader(batch_size)
  train_length = len(train_loader.dataset)
  test_loader = get_test_loader(batch_size)
  test_length = len(test_loader.dataset)

  return (train_loader, train_length), (test_loader, test_length)

def load_config(config_file):
  with open(config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
  return config

def make_student_save_name(save_to, config):
  return os.path.join(save_to, config['classifier']+'_student')


def main():
  args = parse_args()
  seed_all(args.seed)
  if args.device is None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
  else:
    device = args.device

  #########
  # Teacher
  teacher_config = load_config(args.teacher_config_file)
  teacher_model, teacher_model_name = get_model(teacher_config, pretrained=True)
  teacher_model.to(device)
  teacher_model.eval()

  #########
  # Student
  student_config = load_config(args.student_config_file)
  student_model, student_model_name = get_model(student_config, pretrained=False)
  # We don't preload from saved models, but rather from pretrained.
  if args.preload_student:
    state_dict = make_student_save_name(args.save_to, student_config) + '.pt'
    state_dict = torch.load(state_dict)
    student_model.load_state_dict(state_dict)
  student_model.to(device)
  student_model.eval()

  ######
  # Data
  if args.batch_size is None:
    args.batch_size = student_config['batch_size']
  (train_loader, train_length), (test_loader, test_length) = get_loaders(args.batch_size)
  print(f'batch_size: {args.batch_size}')

  ############################################
  # Initial run to make sure the models are OK
  teacher_num_correct = 0
  student_num_correct = 0
  with torch.no_grad():
    for img, lbl in test_loader:
      img = img.to(device)
      lbl = lbl.to(device)
      teacher_pred = teacher_model(img).argmax(-1)
      teacher_num_correct += (teacher_pred == lbl).float().sum()
      student_pred = student_model(img).argmax(-1)
      student_num_correct += (student_pred == lbl).float().sum()
  teacher_accuracy = teacher_num_correct / test_length
  student_accuracy = student_num_correct / test_length
  print(f'===> Accuracy of the teacher model {teacher_model_name} is {teacher_accuracy:.2%}')
  print(f'===> Accuracy of the student model {student_model_name} is {student_accuracy:.2%}')

  ############
  # Optimizers
  student_config['scheduler_kwargs']['steps_per_epoch'] = ceil(train_length / args.batch_size)
  optimizer, scheduler = get_optimizer(model=student_model, config=student_config)

  ################################
  # Store the history for plotting
  history = {
    'teacher_name': teacher_config['classifier'],
    'student_name': student_config['classifier'],
    'epochs': [],
    'train': {
      'accuracy': [],
      'loss': []
    },
    'test': {
      'accuracy': [],
      'loss': []
    }
  }

  ###########################
  # Run the self-distillation
  for epoch in range(student_config['max_epochs']):
    print(f'Epoch {epoch+1}/{student_config["max_epochs"]}')
    history['epochs'].append(epoch)

    # ===> Train
    epoch_loss, epoch_accuracy = epoch_self_distillation_train(
      teacher_model=teacher_model,
      student_model=student_model,
      loader=train_loader,
      loss_fn=soft_logloss,
      correct_fn=num_correct_fn,
      optimizer=optimizer,
      scheduler=scheduler,
      device=device
    )

    history['train']['loss'].append(epoch_loss)
    history['train']['accuracy'].append(epoch_accuracy)
    print(f'\t- Training Loss: {epoch_loss:.2e}, Accuracy: {epoch_accuracy:.2%}')

    ## ===> Test
    epoch_loss, epoch_accuracy = epoch_self_distillation_test(
      teacher_model=teacher_model,
      student_model=student_model,
      loader=test_loader,
      loss_fn=soft_logloss,
      correct_fn=num_correct_fn,
      device=device
    )

    history['test']['loss'].append(epoch_loss)
    history['test']['accuracy'].append(epoch_accuracy)
    print(f'\t- Test Loss: {epoch_loss:.2e}, Accuracy: {epoch_accuracy:.2%}')

  ##############
  # Save results
  os.makedirs(args.save_to, exist_ok=True)
  json_path = make_student_save_name(args.save_to, student_config) + '.json'
  with open(json_path, 'w') as f:
    json.dump(history, f)

  model_save_path = make_student_save_name(args.save_to, student_config) + '.pt'
  torch.save(student_model.state_dict(), model_save_path)

if __name__ == '__main__':
  main()
