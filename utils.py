import os
import torch
import logging
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


def restore_checkpoint(ckpt_dir, state, device):
  if not os.path.exists(ckpt_dir):
    # os.makedirs(ckpt_dir, exist_ok=True)
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    try:
      state['epoch'] = loaded_state['epoch']
    except: pass
    return state


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step'],
    'epoch': state['epoch'],
  }
  torch.save(saved_state, ckpt_dir)



def apply_activate(data, output_info):
    data_t = []
    st = 0
    for item in output_info:
        if item[1] == 'tanh':
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        elif item[1] == 'sigmoid':
            ed = st + item[0]
            data_t.append(data[:,st:ed])
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            data_t.append(F.softmax(data[:, st:ed]))

            st = ed
        else:
            assert 0
    return torch.cat(data_t, dim=1)
