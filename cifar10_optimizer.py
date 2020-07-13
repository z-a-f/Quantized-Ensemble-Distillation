
from torch import optim

def get_optimizer(model, config):
  optimizer_class = getattr(optim, config['optimizer'])
  optimizer = optimizer_class(model.parameters(), **config['optimizer_kwargs'])

  scheduler_class = getattr(optim.lr_scheduler, config['scheduler'])
  scheduler = scheduler_class(optimizer, epochs=config['max_epochs'], **config['scheduler_kwargs'])

  return optimizer, scheduler
