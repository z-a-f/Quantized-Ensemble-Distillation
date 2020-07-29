import argparse
import copy
import os
from math import ceil
import sys
import yaml
import json

import numpy as np
import torch
import torch.quantization as tq

from cifar10_data import get_train_loader, get_test_loader
from cifar10_loss import soft_logloss, num_correct as num_correct_fn
from cifar10_optimizer import get_optimizer
from cifar10_paths import models_path, state_dict_path
from cifar10_pretrained import get_model, get_qat_model
from cifar10_run import epoch_self_distillation_train
from cifar10_run import epoch_self_distillation_test

import torch.quantization._numeric_suite as ns

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('teacher_config_file', type=str, help='YAML configuration file for the teacher')
  parser.add_argument('student_config_file', type=str, help='YAML configuration file for the student')
  parser.add_argument('--preload_student', action='store_const', const=True, default=False)
  parser.add_argument('--qat', action='store_const', const=True, default=False)
  parser.add_argument('--seed', type=int, default=None, help='Repro seed')
  parser.add_argument('--device', type=str, default=None, help='Force device')
  parser.add_argument('--batch-size', type=int, default=None)
  parser.add_argument('--save-to', type=str, default='results')

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

def load_student_model(config, *, qat=False, preload_state_dict=False, preload_dir='results'):
  if qat:
    student_model, student_model_name = get_qat_model(config, pretrained=False)
  else:
    student_model, student_model_name = get_model(config, pretrained=False)
  # We don't preload from saved models, but rather from pretrained.
  if preload_state_dict:
    state_dict = make_student_save_name(preload_dir, config) + '.pt'
    state_dict = torch.load(state_dict)
    student_model.load_state_dict(state_dict)
  if qat:
    if hasattr(student_model, 'fuse_model'):
      student_model.fuse_model()
    student_model.qconfig = tq.get_default_qat_qconfig('fbgemm')
    tq.prepare_qat(student_model, inplace=True)

  return student_model, student_model_name

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
  student_model, student_model_name = load_student_model(
    student_config, qat=args.qat, preload_state_dict=args.preload_student,
    preload_dir=args.save_to)
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
  teacher_accuracy = (teacher_num_correct / test_length).item()
  student_accuracy = (student_num_correct / test_length).item()
  print(f'===> Initial accuracy of the teacher model {teacher_model_name} is {teacher_accuracy:.2%}')
  print(f'===> Initial accuracy of the student model {student_model_name} is {student_accuracy:.2%}')

  ############
  # Optimizers
  student_config['scheduler_kwargs']['steps_per_epoch'] = ceil(train_length / args.batch_size)
  optimizer, scheduler = get_optimizer(model=student_model, config=student_config)

  ################################
  # Store the history for plotting
  history = {
    'teacher_name': teacher_config['classifier'],
    'teacher_max_accuracy': teacher_accuracy,
    'student_name': student_config['classifier'],
    'student_max_accuracy': 0.0,
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

    if args.qat:
      pretrain_for = student_config.get('qat', {'pretrain_epochs': 10})
      pretrain_for = pretrain_for['pretrain_epochs']
      if epoch < pretrain_for:
        # Due to a bug in quantization (#41791), we cannot just use QAT on a
        # newly created model -- need to pretrain it a little before enabling
        # the QAT.
        student_model.apply(tq.disable_fake_quant)
        student_model.apply(tq.disable_observer)
        student_model.apply(torch.nn.intrinsic.qat.update_bn_stats)
      elif pretrain_for <= epoch <= pretrain_for + 2:  # Calibrate
        student_model.apply(tq.enable_fake_quant)
        student_model.apply(tq.enable_observer)
        student_model.apply(torch.nn.intrinsic.qat.update_bn_stats)
      elif epoch > pretrain_for + 2:  # Start QAT
        student_model.apply(tq.disable_observer)
        student_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

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

    if False and args.qat:
      student_model_ = copy.deepcopy(student_model).to('cpu')
      tq.convert(student_model_.eval(), inplace=True)
    else:
      student_model_ = student_model
    ## ===> Test
    epoch_loss, epoch_accuracy = epoch_self_distillation_test(
      teacher_model=teacher_model,
      student_model=student_model_,
      loader=test_loader,
      loss_fn=soft_logloss,
      correct_fn=num_correct_fn,
      device=device
    )

    history['test']['loss'].append(epoch_loss)
    history['test']['accuracy'].append(epoch_accuracy)
    print(f'\t- Test Loss: {epoch_loss:.2e}, Accuracy: {epoch_accuracy:.2%}')
  history['student_max_accuracy'] = max(history['test']['accuracy'])

  ##############
  # Save results
  os.makedirs(args.save_to, exist_ok=True)
  file_name = make_student_save_name(args.save_to, student_config)
  file_name += ('_qat' if args.qat else '')
  json_path = file_name + '.json'
  with open(json_path, 'w') as f:
    json.dump(history, f)

  model_save_path = file_name + '.pt'
  torch.save(student_model.state_dict(), model_save_path)

if __name__ == '__main__':
  main()
