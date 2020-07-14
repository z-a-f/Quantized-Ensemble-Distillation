import torch

def epoch_self_distillation_train(teacher_model, student_model, loader, loss_fn,
                                  correct_fn, optimizer, scheduler, device):
  running_correct = 0.0
  running_loss = 0.0
  set_length = len(loader.dataset)

  teacher_model.eval()
  student_model.train()

  for x, y in loader:
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
      teacher_predictions = teacher_model(x).softmax(-1)

    student_logits = student_model(x)
    loss = loss_fn(student_logits, teacher_predictions)

    running_loss += loss.item()
    running_correct += correct_fn(student_logits, y).item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

  return running_loss / set_length, running_correct / set_length

def epoch_self_distillation_test(teacher_model, student_model, loader, loss_fn,
                                 correct_fn, device):
  running_correct = 0.0
  running_loss = 0.0
  set_length = len(loader.dataset)

  teacher_model.eval()  # Only need that for the loss computation
  student_model.eval()

  with torch.no_grad():
    for x, y in loader:
      x = x.to(device)
      y = y.to(device)

      teacher_predictions = teacher_model(x).softmax(-1)
      student_logits = student_model(x)
      loss = loss_fn(student_logits, teacher_predictions)

      running_loss += loss.item()
      running_correct += correct_fn(student_logits, y).item()


  return running_loss / set_length, running_correct / set_length
