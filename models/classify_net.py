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

from . import layers_cls,layers
import torch.nn as nn
import torch
from torch.nn.functional import embedding
from . import utils, layers, layerspp

default_initializer = layers_cls.default_init

NONLINEARITIES = {
    "elu": nn.ELU(),
    "relu": nn.ReLU(),
    "lrelu": nn.LeakyReLU(negative_slope=0.2),
    "swish": nn.SiLU(),
    "tanh": nn.Tanh(),
    "softplus": nn.Softplus(),
  "selu":nn.SELU()
}

class tabularUnet(nn.Module):
    '''
        arg = {
            't_embed_dim' : 16,
            'cond_size' : size of cond,
            'cond_out_size' : size of cond out,
            'encoder_dim' : [64,128,256],  # the embedding space
            'activation' : 'relu',
            'output_size' : the predict size of modle

        };
        time_embed return 2 dim tensor
    '''
    def __init__(self, config):
        arg = {
            't_embed_dim': config.model.nf,
            'cond_size': config.data.image_size,
            'cond_out_size': config.data.image_size,  #
            'encoder_dim': list(config.model.hidden_dims)[:len(config.model.hidden_dims) // 2],  # the embedding space
            # 'encoder_dim':[64, 128, 256],
            'activation': config.model.activation,
            'output_size': config.data.output_size  # the dim of y
        }
        super().__init__()

        self.embed_dim = arg['t_embed_dim']
        tdim = self.embed_dim * 4
        self.act = layers_cls.get_act(arg['activation'])

        # create two layer and initializer for time embedding
        modules = []
        modules.append(nn.Linear(self.embed_dim, tdim))
        modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)
        modules.append(nn.Linear(tdim, tdim))
        modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)
        # condition layer
        cond = arg['cond_size']
        cond_out = (arg['cond_out_size'])   # tdim =?

        modules.append(nn.Linear(cond, cond_out))
        modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)

        self.all_modules = nn.ModuleList(modules)
        # this layer for x_0 + condition
        dim_in = cond_out
        dim_out = arg['encoder_dim'][0]
        self.inputs = nn.Linear(dim_in, dim_out)  # input layer

        self.encoder = layers_cls.Encoder(arg['encoder_dim'], tdim, arg['activation'])  # encoder

        dim_in = arg['encoder_dim'][-1]
        dim_out = arg['output_size']
        self.bottom_block = nn.Linear(dim_in, dim_out)  # bottom_layer


    
    def forward(self,cond, time_cond ):
        modules = self.all_modules
        m_idx = 0

        # time embedding
        temb = layers_cls.get_timestep_embedding(time_cond, self.embed_dim)
        temb = modules[m_idx](temb)
        m_idx += 1
        temb = self.act(temb)
        temb = modules[m_idx](temb)
        m_idx += 1

        # condition layer
        cond = modules[m_idx](cond.float())
        m_idx += 1

        inputs = self.inputs(cond)  # input layer
        skip_connections, encoding = self.encoder(inputs, temb)
        encoding = self.bottom_block(encoding)
        encoding = self.act(encoding)

        return encoding



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
    }

    self.config = config
    self.act = layers.get_act(config)
    self.hidden_dims = config.model.hidden_dims


    modules = []
    dim = config.data.image_size
    for item in list(config.model.hidden_dims):
      modules += [
          base_layer[config.model.class_layer_type](dim, item)
      ]
      dim += item  # residual part
      # dim = item  # for del the residual part
      modules.append(NONLINEARITIES[config.model.activation])
      if config.training.dropout:  # add drop layer
        modules.append(nn.Dropout(p=0.2))

    modules.append(nn.Linear(dim, config.data.output_size))
    self.all_modules = nn.ModuleList(modules)

  def forward(self, x, time_cond):
    if len(x.shape) ==1:  # if x only the input
      x = x.view(1,-1)
    modules = self.all_modules
    m_idx = 0

    temb = x
    for _ in range(len(self.hidden_dims)):
      temb1 = modules[m_idx](t=time_cond, x=temb)
      temb = torch.cat([temb1, temb], dim=1)
      # temb = temb1  # for del the residual part
      m_idx += 1
      temb = modules[m_idx](temb)
      m_idx += 1
      if self.config.training.dropout:  # add drop layer
        temb = modules[m_idx](temb)
        m_idx += 1
    h = modules[m_idx](temb)


    return h

if __name__ == '__main__':
    arg = {
            't_embed_dim' : 16,
            'cond_size' : 5,
            'cond_out_size' : 5,   # 
            'encoder_dim' : [64,128,256] ,# the embedding space
            'activation' : 'relu',
            'output_size' : 5
        }
    model = tabularUnet(arg)
    x = torch.tensor([[1,2,3,4,5],])
    t = torch.tensor([1])
    print(model(x,t))
