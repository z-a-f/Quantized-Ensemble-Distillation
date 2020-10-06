from math import ceil
import json
import os
import yaml

import torch
from torch import optim
from torch import nn

from cifar10_data import get_train_loader, get_test_loader
from cifar10_pretrained import get_model
from cifar10_loss import soft_logloss, num_correct

from qat_models.lenet5 import LeNet5Ensemble

device = 'cuda'
batch_size = 1024
EPOCHS = 100
ensemble_size = 10

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

def ensemble_loss_fn(loss_fn):
  def _loss_fn(y_hat_list, y):
    losses = []
    for y_hat in y_hat_list:
      losses.append(loss_fn(y_hat, y))
    return sum(losses) / len(losses)
  return _loss_fn


#########
# Teacher
teacher_config = load_config('config/mobilenetv2.yaml')
teacher_model, teacher_model_name = get_model(teacher_config, pretrained=True)
teacher_model.to(device)
teacher_model.eval()

#########
# Student
student_models = LeNet5Ensemble(ensemble_size)
student_models.to(device)
student_models.eval()
student_model_name = 'LeNet5 Ensemble'

######
# Data
if batch_size is None:
  batch_size = student_config['batch_size']
(train_loader, train_length), (test_loader, test_length) = get_loaders(batch_size)
print(f'batch_size: {batch_size}')

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
    student_pred = student_models(img, reduce_=True).argmax(-1)
    student_num_correct += (student_pred == lbl).float().sum()
teacher_accuracy = (teacher_num_correct / test_length).item()
student_accuracy = (student_num_correct / test_length).item()
print(f'===> Initial accuracy of the teacher model {teacher_model_name} is {teacher_accuracy:.2%}')
print(f'===> Initial accuracy of the student model {student_model_name} is {student_accuracy:.2%}')

############
# Optimizers
lr = 1e-2
# optimizer = optim.SGD(student_models.models.parameters(), lr=lr, weight_decay=0.01, momentum=0.9, nesterov=True)
# scheduler = optim.lr_scheduler.OneCycleLR(
#   optimizer, steps_per_epoch=ceil(train_length / batch_size), epochs=EPOCHS, max_lr=lr)
optimizer = optim.Adam(student_models.parameters(), lr=lr)
loss_fn = ensemble_loss_fn(soft_logloss)

##############
# Run training
history = {
  'teacher_name': 'MobileNetV2',
  'teacher_max_accuracy': teacher_accuracy,
  'student_name': f'Lenet5 Ensemble ({len(student_models.models)}x)',
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
max_accuracy = 0.0
os.makedirs('results', exist_ok=True)
file_name = f'results/lenet5_ensemble{len(student_models.models)}'

teacher_model.eval()
for epoch in range(EPOCHS):
  print(f'Epoch {epoch+1}/{EPOCHS}')
  history['epochs'].append(epoch)

  running_correct = 0.0
  running_loss = 0.0
  student_models.train()
  for x, y in train_loader:
    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
      teacher_predictions = teacher_model(x).softmax(-1)

    student_logits = student_models(x, reduce_=False)
    loss = loss_fn(student_logits, teacher_predictions)

    running_loss += loss.item()
    running_correct += num_correct(student_models.reduce_fn(student_logits), y).item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # scheduler.step()

  epoch_loss = running_loss / train_length
  epoch_accuracy = running_correct / train_length
  history['train']['loss'].append(epoch_loss)
  history['train']['accuracy'].append(epoch_accuracy)
  print(f'\t- Training Loss: {epoch_loss:.2e}, Accuracy: {epoch_accuracy:.2%}')

  running_correct = 0.0
  running_loss = 0.0
  student_models.eval()
  with torch.no_grad():
    for x, y in test_loader:
      x = x.to(device)
      y = y.to(device)

      teacher_predictions = teacher_model(x).softmax(-1)

      student_logits = student_models(x, reduce_=False)
      loss = loss_fn(student_logits, teacher_predictions)

      running_loss += loss.item()
      running_correct += num_correct(student_models.reduce_fn(student_logits), y).item()

  epoch_loss = running_loss / test_length
  epoch_accuracy = running_correct / test_length
  history['test']['loss'].append(epoch_loss)
  history['test']['accuracy'].append(epoch_accuracy)
  print(f'\t- Test Loss: {epoch_loss:.2e}, Accuracy: {epoch_accuracy:.2%}')

  # Save the models if it's the best now
  if max_accuracy <= epoch_accuracy:
    max_accuracy = epoch_accuracy
    torch.save(student_models.state_dict(), file_name + '.pt')

history['student_max_accuracy'] = max(history['test']['accuracy'])
with open(file_name + '.json', 'w') as f:
  json.dump(history, f)
