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

# Lint as: python3

from configs.default_tabular_configs import get_default_configs


def get_config():
  config = get_default_configs()
  config.data.dataset = "contraceptive"
  config.data.image_size = 9
  config.data.output_size = 3
  config.training.label = False
  config.training.encoder = 'one-hot'

  config.training.batch_size = 500
  config.eval.batch_size = 500
  # fine
  config.cosine_factor = 4
  config.filter_factor = 1.468640736
  config.time_step = 73
  config.classify_lambda = 46
  config.k_nbrs = 10
  config.easy_fliter = 0.342613799
  config.best_value = 0.480929777
  # training
  training = config.training
  training.sde = 'vesde'
  training.continuous = True
  training.reduce_mean = True
  training.n_iters = 100000
  training.tolerance = 1e-03
  training.hutchinson_type = "Rademacher"
  training.retrain_type = "median"
  
  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'ancestral_sampling'
  sampling.corrector = 'none'

  # model
  model = config.model
  model.layer_type = 'concatsquash'
  model.name = 'ncsnpp_tabular'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.activation = 'selu'

  model.nf = 64
  model.hidden_dims = (64,64,64,64)
  model.conditional = True
  model.embedding_type = 'fourier'
  model.fourier_scale = 16
  model.conv_size = 3

  model.sigma_min = 0.01
  model.sigma_max = 10.

  # test
  test = config.test
  test.n_iter = 1

  # optim
  optim = config.optim
  optim.lr = 5e-4

  
  return config