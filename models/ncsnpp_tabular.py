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

# pylint: skip-file

from torch.nn.functional import embedding
from . import utils, layers, layerspp
import torch.nn as nn
import torch

get_act = layers.get_act
default_initializer = layers.default_init


NONLINEARITIES = {
    "elu": nn.ELU(),
    "relu": nn.ReLU(),
    "lrelu": nn.LeakyReLU(negative_slope=0.2),
    "swish": nn.SiLU(),
    "tanh": nn.Tanh(),
    "softplus": nn.Softplus(),
  "selu":nn.SELU()
}


@utils.register_model(name='ncsnpp_tabular')
class NCSNpp(nn.Module):
  """NCSN++ model"""

  def __init__(self, config):
    super().__init__()
    base_layer = {
      "ignore": layers.IgnoreLinear,
      "squash": layers.SquashLinear,
      "concat": layers.ConcatLinear,
      "concat_v2": layers.ConcatLinear_v2,
      "concatsquash": layers.ConcatSquashLinear,
      "blend": layers.BlendLinear,
      "concatcoord": layers.ConcatLinear,
      "concatsquash_condition_y": layers.ConcatSquashConditionYLinear

    }

    self.config = config
    self.act = get_act(config)
    self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))
    self.hidden_dims = config.model.hidden_dims 


    modules = []
    dim = config.data.image_size
    for item in list(config.model.hidden_dims):
      modules += [
          base_layer[config.model.layer_type](dim, item)
      ]
      # dim += item
      dim = item  # for del the residual part
      modules.append(NONLINEARITIES[config.model.activation])
      if config.training.dropout:  # add drop layer
        modules.append(nn.Dropout(p=0.2))

    modules.append(nn.Linear(dim, config.data.image_size))
    self.all_modules = nn.ModuleList(modules)

  def forward(self, x, time_cond):
    if len(x.shape) ==1:  # if x only the input
      x = x.view(1,-1)
    modules = self.all_modules 
    m_idx = 0

    temb = x
    for _ in range(len(self.hidden_dims)):
      temb1 = modules[m_idx](t=time_cond, x=temb)
      # temb = torch.cat([temb1, temb], dim=1)
      temb = temb1  # for del the residual part
      m_idx += 1
      temb = modules[m_idx](temb) 
      m_idx += 1
      if self.config.training.dropout:  # add drop layer
        temb = modules[m_idx](temb)
        m_idx += 1
    h = modules[m_idx](temb)


    return h

@utils.register_model(name='ncsnpp_tabular_condition_y')
class NCSNppCY(nn.Module):
  """NCSN++ model"""

  def __init__(self, config):
    super().__init__()
    base_layer = {
      "ignore": layers.IgnoreLinear,
      "squash": layers.SquashLinear,
      "concat": layers.ConcatLinear,
      "concat_v2": layers.ConcatLinear_v2,
      "concatsquash": layers.ConcatSquashLinear,
      "blend": layers.BlendLinear,
      "concatcoord": layers.ConcatLinear,
      "concatsquash_condition_y": layers.ConcatSquashConditionYLinear

    }

    self.config = config
    self.act = get_act(config)
    self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))
    self.hidden_dims = config.model.hidden_dims
    outputsize = config.data.output_size if config.training.encoder == 'one-hot' else 1

    modules = []
    dim = config.data.image_size
    for item in list(config.model.hidden_dims):
      modules += [
          base_layer[config.model.layer_type](dim, item, outputsize)
      ]
      # dim += item
      dim = item  # for del the residual part
      modules.append(NONLINEARITIES[config.model.activation])
      if config.training.dropout:  # add drop layer
        modules.append(nn.Dropout(p=0.2))

    modules.append(nn.Linear(dim, config.data.image_size))
    self.all_modules = nn.ModuleList(modules)

  def forward(self, x, time_cond, y_cond):
    if len(x.shape) ==1:  # if x only the input
      x = x.view(1,-1)
    modules = self.all_modules
    m_idx = 0

    temb = x
    for _ in range(len(self.hidden_dims)):
      temb1 = modules[m_idx](t=time_cond, x=temb, y = y_cond)
      # temb = torch.cat([temb1, temb], dim=1)
      temb = temb1  # for del the residual part
      m_idx += 1
      temb = modules[m_idx](temb)
      m_idx += 1
      if self.config.training.dropout:  # add drop layer
        temb = modules[m_idx](temb)
        m_idx += 1

    h = modules[m_idx](temb)


    return h
