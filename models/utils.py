# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions and modules related to model definition.
"""

import torch
import sde_lib
import numpy as np
import random

_MODELS = {}


def register_model(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _MODELS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _MODELS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_model(name):
  return _MODELS[name]


def get_sigmas(config):
  """Get sigmas --- the set of noise levels for SMLD from config files.
  Args:
    config: A ConfigDict object parsed from the config file
  Returns:
    sigmas: a jax numpy arrary of noise levels
  """
  sigmas = np.exp(
    np.linspace(np.log(config.model.sigma_max), np.log(config.model.sigma_min), config.model.num_scales))

  return sigmas


def get_ddpm_params(config):
  num_diffusion_timesteps = 1000
  beta_start = config.model.beta_min / config.model.num_scales
  beta_end = config.model.beta_max / config.model.num_scales
  betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

  alphas = 1. - betas
  alphas_cumprod = np.cumprod(alphas, axis=0)
  sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
  sqrt_1m_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

  return {
    'betas': betas,
    'alphas': alphas,
    'alphas_cumprod': alphas_cumprod,
    'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
    'sqrt_1m_alphas_cumprod': sqrt_1m_alphas_cumprod,
    'beta_min': beta_start * (num_diffusion_timesteps - 1),
    'beta_max': beta_end * (num_diffusion_timesteps - 1),
    'num_diffusion_timesteps': num_diffusion_timesteps
  }


def create_model(config):
  """Create the score model."""
  model_name = config.model.name
  score_model = get_model(model_name)(config)
  score_model = score_model.to(config.device)
  score_model = torch.nn.DataParallel(score_model)
  return score_model


def get_model_fn(model, train=False, cond_y=False, config = None):
  """Create a function to give the output of the score-based model.

  Args:
    model: The score model.
    train: `True` for training and `False` for evaluation.

  Returns:
    A model function.
  """

  def model_fn(x, labels):
    """Compute the output of the score-based model.

    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.

    Returns:
      A tuple of (model output, new mutable states)
    """
    if not train:
      # model.eval()  # temporarily ban the function, if there exist the drop out or batch normalizition
      pass
    else:
      model.train()
    if cond_y:

      x1 = x[:, :-config.data.output_size].float()
      y = x[:, -config.data.output_size:]
      return model(x1, labels, y)
    else:
      return model(x, labels)

  return model_fn


def get_score_fn(sde, model, train=False, continuous=False, cond_y=False,config=None):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.

  Returns:
    A score function.
  """
  model_fn = get_model_fn(model, train=train, cond_y=cond_y,config=config)

  if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
    def score_fn(x, t):
      if continuous or isinstance(sde, sde_lib.subVPSDE):
        labels = t * (sde.N - 1)
        score = model_fn(x, labels)
        std = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        labels = t * (sde.N - 1)
        score = model_fn(x, labels) 
        std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]
      score = -score / std[:, None]

      return score

  elif isinstance(sde, sde_lib.VESDE):
    def score_fn(x, t):
      if continuous:
        labels = sde.marginal_prob(torch.zeros_like(x), t)[1]   # scale the t
      else:
        labels = sde.T - t
        labels *= sde.N - 1
        labels = torch.round(labels).long()

      score = model_fn(x, labels)
      return score

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn


def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))

def after_defined(flags,config):
  if flags.cond_y:    # for p(x|y)
    config.training.label = True  # train classify always need label, but the xscore different
    config.model.layer_type = 'concatsquash_condition_y'
    config.model.name = 'ncsnpp_tabular_condition_y'
    config.training.cond_y = True
  else:    # for p(x)
    config.training.label = False  # train classify always need label, but the xscore different
    config.model.layer_type = 'concatsquash'
    config.model.name = 'ncsnpp_tabular'
    config.training.cond_y = False
  if flags.xscore_label: # for p(x,y)
    config.data.image_size += 1
    config.training.label = True  # train classify always need label, but the xscore different
    config.training.xscore_label = True

def add_label_noise(labels, config):
  # labels's type is np array, return a tensor, and shuffle is ok when load data because the seed
  noise_index = [0 for i in range(len(labels))]
  # 计算每个类别20%的噪声样本数量
  indices_to_modify = []
  labels_numer = np.argmax(labels, axis=1)
  for class_label in np.unique(labels_numer):
    # 获取当前类别的所有索引
    class_indices = np.where(labels_numer == class_label)[0]
    # 计算要加入噪声的数量 (四舍五入到最近的整数)
    num_to_modify = int(np.ceil(config.training.noise * len(class_indices)))
    # 随机选择这些索引
    selected_indices = np.random.choice(class_indices, num_to_modify, replace=False)
    indices_to_modify.extend(selected_indices)

  for n_i in indices_to_modify:
    noise_index[n_i] = 1

  noise_label_ohe = labels[indices_to_modify]
  noise_label = labels_numer[indices_to_modify]
  add_ = [random.choice([i for i in range(1, config.data.output_size)]) for i in range(len(noise_label))]
  noise_label = (noise_label+ add_)%config.data.output_size
  zero = np.zeros_like(noise_label_ohe)
  zero[np.arange(len(zero)),noise_label] = 1
  labels[indices_to_modify] = zero
  return labels, noise_index
