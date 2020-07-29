import torch
from torch import nn

class LeNet5(nn.Module):
  r"""This is a modified LeNet5 model:
    Some changes were required to allow the QAT.
    TanH is replaced with ReLU -- PTQ/QAT don't support tanh yet
    We could reuse the ReLU, but we wouldn't be able to quantize it in that case
    Also, added adaptive average pooling to support different size inputs.
  """

  def __init__(self):
    super(LeNet5, self).__init__()
    self.C1 = nn.Conv2d(3, 6, kernel_size=5, padding=0, stride=1)
    self.A1 = nn.ReLU(inplace=True)
    self.S1 = nn.MaxPool2d(2, padding=0, stride=2)
    self.C2 = nn.Conv2d(6, 16, kernel_size=5, padding=0, stride=1)
    self.A2 = nn.ReLU(inplace=True)
    self.S2 = nn.MaxPool2d(2, padding=0, stride=2)
    self.C3 = nn.Conv2d(16, 120, kernel_size=5, padding=0, stride=1)
    self.A3 = nn.ReLU(inplace=True)

    self.PoolTo1x1 = nn.AdaptiveAvgPool2d((1, 1))
    self.Flatten = nn.Flatten(start_dim=1)

    self.FC1 = nn.Linear(120, 84)
    self.FC_A1 = nn.ReLU(inplace=True)
    self.FC2 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.C1(x)
    x = self.A1(x)
    x = self.S1(x)
    x = self.C2(x)
    x = self.A2(x)
    x = self.S2(x)
    x = self.C3(x)

    x = self.PoolTo1x1(x)
    x = self.Flatten(x)

    x = self.FC1(x)
    x = self.FC_A1(x)
    x = self.FC2(x)
    return x


class LeNet5Ensemble(nn.Module):
  def __init__(self, ensemble_size=10, reduce_fn=None):
    super(LeNet5Ensemble, self).__init__()
    self.models = nn.ModuleList()
    self.reduce_fn = reduce_fn
    if self.reduce_fn is None:
      self.reduce_fn = lambda x: torch.stack(x, 0).mean(0)
    for idx in range(ensemble_size):
      self.models.append(LeNet5())

  def forward(self, x, *, reduce_=False):
    logits = []
    for model in self.models:
      logits.append(model(x))
    if reduce_:
      logits = self.reduce_fn(logits)
    return logits

