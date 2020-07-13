
import os
from math import ceil
import sys
import yaml

import numpy as np
import torch

from cifar10_data import get_train_loader, get_test_loader
from cifar10_loss import soft_logloss, num_correct as num_correct_fn
from cifar10_optimizer import get_optimizer
from cifar10_paths import models_path, state_dict_path, student_save_path
from cifar10_pretrained import get_model
from cifar10_run import epoch_self_distillation_train
from cifar10_run import epoch_self_distillation_test

#######
# Repro
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

###########
# Variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 512

train_loader = get_train_loader(batch_size)
train_length = len(train_loader.dataset)
test_loader = get_test_loader(batch_size)
test_length = len(test_loader.dataset)

#######
# Model
config_file = 'mobilenetv2.yaml'
with open(config_file, 'r') as f:
  config = yaml.load(f, Loader=yaml.FullLoader)

teacher_model, teacher_model_name = get_model(config, pretrained=True)
teacher_model = teacher_model.to(device)
teacher_model.eval()

##########################
# Run inference on teacher
num_correct = 0
with torch.no_grad():
  for img, lbl in test_loader:
    img = img.to(device)
    lbl = lbl.to(device)
    pred = teacher_model(img).argmax(-1)
    num_correct += (pred == lbl).float().sum()
accuracy = num_correct / test_length
print(f'===> Accuracy of the loaded teacher model {teacher_model_name} is {accuracy:.2%}')

##############################################
# Create a student model for self-distillation
student_model, student_model_name = get_model(config, pretrained=False)
student_model.train()
student_model = student_model.to(device)

config['scheduler_kwargs']['steps_per_epoch'] = ceil(train_length / batch_size)
optimizer, scheduler = get_optimizer(model=student_model, config=config)

history = {
  'teacher_name': config['classifier'],
  'student_name': config['classifier'],
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
for epoch in range(config['max_epochs']):
  print(f'Epoch {epoch+1}/{config["max_epochs"]}')
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

########################################
# Save the results and the student model
import json
with open('results_' + config['classifier'] + '.json', 'w') as f:
  json.dump(history, f)

student_save_path = os.path.join(student_save_path, config['classifier']+'_student.pt')
torch.save(student_model.state_dict(), student_save_path)
