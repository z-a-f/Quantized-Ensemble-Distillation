
import torch

def soft_logloss(y_hat, y):
  logprobs = torch.nn.functional.log_softmax(y_hat, -1)
  return -(y * logprobs).sum() / len(y)

def num_correct(y_hat_logits, y):
  return (y_hat_logits.argmax(-1) == y.data).float().sum()
