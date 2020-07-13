
import argparse
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

rc_params = {
  'font': {
    'family'     : 'monospace',
    'style'      : 'normal',
    'variant'    : 'normal',
    'weight'     : 'bold',
    'stretch'    : 'normal',
    'size'       : 18,
  },
  'text': {'usetex'     : True,},
  'xtick': {'labelsize' : 18,},
  'ytick': {'labelsize' : 18,},
  'axes': {
    'labelsize'  : 18,
    'titlesize'  : 18,
  },
  'lines': {
    'linewidth' : 2,
    'markersize': 6,
  },
}

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('json_file', type=str)
  parser.add_argument('--save_to', type=str, default=None)

  return parser.parse_args()

def plot(data):
  for key, param in rc_params.items():
    mpl.rc(key, **param)
  # Collect plottable keys
  modes = []
  # y = []
  for key, value in data.items():
    if key == 'epoch' or key == 'epochs':
      epochs = value
    elif isinstance(value, dict):
      modes.append(key)
  #   elif isinstance(value, (list, tuple)):
  #     y.append(key)
  # assert len(y) == 0 or len(modes) == 0
  fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True)

  for mode in modes:
    ax[0].plot(epochs, data[mode]['loss'], label=mode)
    ax[1].plot(epochs, data[mode]['accuracy'], label=mode)

  ax[0].set_xlabel('Epoch')
  ax[0].set_ylabel('Loss')
  ax[0].grid()
  ax[0].legend()

  ax[1].set_xlabel('Epoch')
  ax[1].set_ylabel('Accuracy')
  ax[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
  ax[1].grid()
  ax[1].legend()

  teacher_name = data['teacher_name'].replace('_', r'\_')
  student_name = data['student_name'].replace('_', r'\_')
  plt.suptitle(f'Teacher: {teacher_name}\nStudent: {student_name}')
  plt.tight_layout()
  # plt.subplots_adjust(top=0.9)
  plt.tight_layout(rect=[0, 0.0, 1, 0.93])

  return ax

def main():
  args = parse_args()
  with open(args.json_file, 'r') as f:
    data = json.load(f)
  ax = plot(data)

  if args.save_to is None:
    plt.show()
  else:
    plt.savefig(args.save_to)

if __name__ == '__main__':
  main()
