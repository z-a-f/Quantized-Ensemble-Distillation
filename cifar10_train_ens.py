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
from cifar10_paths import models_path, state_dict_path, student_save_path
from cifar10_pretrained import get_model
from cifar10_run import epoch_self_distillation_train
from cifar10_run import epoch_self_distillation_test

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('teacher_config_file', type=str, help='YAML configuration file for the teacher')
  parser.add_argument('student_config_file', type=str, help='YAML configuration file for the student')
  parser.add_argument('--seed', type=int, default=None, help='Repro seed')
  parser.add_argument('--device', type=str, default=None, help='Force device')
  parser.add_argument('--batch_size', type=int, default=512)
  parser.add_argument('--save_to', type=str, default='.')

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

def main():
  args = parse_args()
  seed_all(args.seed)
  if args.device is None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
  else:
    device = args.device
  (train_loader, train_length), (test_loader, test_length) = get_loaders(args.batch_size)

  # Teacher
  teacher_config = load_config(args.teacher_config_file)
  teacher_model, teacher_model_name = get_model(teacher_config, pretrained=True)
  teacher_model.to(device)
  teacher_model.eval()

  # Run the teacher model to get the baseline accuracy
  num_correct = 0
  with torch.no_grad():
    for img, lbl in test_loader:
      img = img.to(device)
      lbl = lbl.to(device)
      pred = teacher_model(img).argmax(-1)
      num_correct += (pred == lbl).float().sum()
  accuracy = num_correct / test_length
  print(f'===> Accuracy of the loaded teacher model {teacher_model_name} is {accuracy:.2%}')

  # Student
  student_config = load_config(args.student_config_file)
  student_model, student_model_name = get_model(student_config, pretrained=False)
  student_model.to(device)
  student_model.train()

  student_config['scheduler_kwargs']['steps_per_epoch'] = ceil(train_length / args.batch_size)
  optimizer, scheduler = get_optimizer(model=student_model, config=student_config)

  # Run the training
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

  # Save results
  results_path = args.save_to
  os.makedirs(results_path, exist_ok=True)
  json_path = os.path.join(results_path, student_config['classifier'] + '_student.json')
  with open(json_path, 'w') as f:
    json.dump(history, f)

  model_save_path = os.path.join(results_path, student_config['classifier']+'_student.pt')
  torch.save(student_model.state_dict(), model_save_path)

if __name__ == '__main__':
  main()
