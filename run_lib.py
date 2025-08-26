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
"""Training and evaluation for score-based generative models. """

import numpy as np
import pandas as pd
import logging
from models import ncsnpp_tabular
import losses
import likelihood
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
from torch.utils.data import DataLoader
#import evaluation
import sde_lib
from absl import flags
import random
import torch
from torch.utils import tensorboard
from sklearn.model_selection import KFold
from utils import save_checkpoint, restore_checkpoint, apply_activate
import collections
import os
from scipy.stats import wasserstein_distance
from geomloss import SamplesLoss
from torch.utils.data import RandomSampler
from fineturne import clean_fn
FLAGS = flags.FLAGS


def calculate_wasserstein(sample, train_ds_,config, bins_num=20):
    with torch.no_grad():
        loss = SamplesLoss(loss="sinkhorn", p=1, blur=.05)
        # if sample.shape[0]%2 ==1: # in case of can't to return 2 dimension
        #     sample = sample[:-1]
        # if train_ds_.shape[0]%2 ==1:
        #     train_ds_ = train_ds_[:-1]
        L = loss(sample, train_ds_).item()
        try:
            L = loss(sample, train_ds_).item()
        except:
            L = np.inf
        # bins = np.linspace(0, 1, bins_num)
        # dis = []
        # for i in range(sample.shape[1]):
        #     t_sample = sample[:, i]
        #     t_sample = (t_sample - t_sample.min()) / (t_sample.max() - t_sample.min())
        #     t_train = train_ds_[:, i]
        #     t_train = (t_train - t_train.min()) / (t_train.max() - t_train.min())
        #     frequency_sample, _ = np.histogram(t_sample, bins=bins)
        #     frequency_sample = np.array(frequency_sample).astype(np.float64)
        #     frequency_sample /= sample.shape[0]
        #     frequency_train, _ = np.histogram(t_train, bins=bins)
        #     frequency_train = np.array(frequency_train).astype(np.float64)
        #     frequency_train /= sample.shape[0]
        #     index = [j for j in range(bins_num - 1)]
        #     # frequency_sample = torch.tensor(frequency_sample).to(config.device)
        #     # frequency_train = torch.tensor(frequency_train).to(config.device)
        #     if not sum(frequency_sample) > 0:
        #         return np.inf
        #     else:
        #         dis.append( wasserstein_distance(index, index, frequency_sample, frequency_train))
    return L

def partial_model(model,label):
    ''' for condition y sample way '''
    def return_fn(batch,t):
        return model(batch,t,label)
    return return_fn

def train(config, workdir):



    tb_dir = os.path.join(workdir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = tensorboard.SummaryWriter(tb_dir)


    # Build data iterators
    train_ds, eval_ds, (transformer, meta) = datasets.get_dataset(config,
                                                                  uniform_dequantization=config.data.uniform_dequantization)
    data = np.concatenate((train_ds, eval_ds), axis=0)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    ckp_fdix = 0

    for train_index, val_index in kf.split(data):
        # Initialize model.
        randomSeed = 2021
        torch.manual_seed(randomSeed)

        torch.cuda.manual_seed(randomSeed)
        torch.cuda.manual_seed_all(randomSeed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(randomSeed)
        random.seed(randomSeed)

        # add noise
        if config.training.label:
            label = train_ds[:,-config.data.output_size:]
            noise,_ = mutils.add_label_noise(label,config)
            train_ds[:,-config.data.output_size:] = np.array(noise)

        score_model = mutils.create_model(config)
        num_params = sum(p.numel() for p in score_model.parameters())
        logging.info(f"the number of parameters:{num_params}")
        ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
        optimizer = losses.get_optimizer(config, score_model.parameters())
        state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0, epoch=0)

        ckp_fdix +=1
        logging.info(f"fold {ckp_fdix}")
        train_ds, eval_ds = data[train_index], data[val_index]


        checkpoint_dir = os.path.join(workdir, "checkpoints")
        checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", f"checkpoint{ckp_fdix}.pth")

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)

        state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
        initial_step = int(state['epoch'])



        if meta['problem_type'] == 'binary_classification':
            metric = 'binary_f1'
        elif meta['problem_type'] == 'regression':
            metric = "r2"
        else:
            metric = 'macro_f1'

        logging.info(f"train shape : {train_ds.shape}")
        logging.info(f"eval.shape : {eval_ds.shape}")

        logging.info(f"batch size: {config.training.batch_size}")
        train_ds_ = torch.tensor(train_ds).float().to(config.device)
        #if metric != "r2" and config.training.label:
        #    logging.info('raw data : {}'.format(collections.Counter(train_ds_[:, -1])))

        #sampler = RandomSampler(train_ds, replacement=True, num_samples=((len(train_ds) // 1000) + 1) * 1000)
        train_iter = DataLoader(train_ds, batch_size=config.training.batch_size)
        eval_iter = iter(DataLoader(eval_ds, batch_size=config.eval.batch_size))  # pytype: disable=wrong-arg-types

        scaler = datasets.get_data_scaler(config)
        inverse_scaler = datasets.get_data_inverse_scaler(config)

        # Setup SDEs
        if config.training.sde.lower() == 'vpsde':
            sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            sampling_eps = 1e-3
        elif config.training.sde.lower() == 'subvpsde':
            sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                                   N=config.model.num_scales)
            sampling_eps = 1e-3
        elif config.training.sde.lower() == 'vesde':
            sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                                N=config.model.num_scales)
            sampling_eps = 1e-5
        else:
            raise NotImplementedError(f"SDE {config.training.sde} unknown.")
        logging.info(score_model)

        optimize_fn = losses.optimization_manager(config)
        continuous = config.training.continuous
        reduce_mean = config.training.reduce_mean
        likelihood_weighting = config.training.likelihood_weighting
        cond_y = config.training.cond_y


        train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                           reduce_mean=reduce_mean, continuous=continuous,
                                           likelihood_weighting=likelihood_weighting, workdir=workdir,
                                           spl=config.training.spl, writer=writer,
                                           alpha0=config.model.alpha0, beta0=config.model.beta0, cond_y=cond_y,config=config)
        eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                          reduce_mean=reduce_mean, continuous=continuous,
                                          likelihood_weighting=likelihood_weighting, workdir=workdir,
                                          spl=config.training.spl, writer=writer,
                                          alpha0=config.model.alpha0, beta0=config.model.beta0)
        # Building sampling functions
        if config.training.snapshot_sampling:
            sampling_shape = (1000 if len(train_ds_)>1000 else len(train_ds_), config.data.image_size)
            sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

        if cond_y: # for sample part
            if config.training.encoder == 'ordinal':
                score_model = partial_model(state['model'], train_ds_[:sampling_shape[0], -1])
            else:
                score_model = partial_model(state['model'], train_ds_[:sampling_shape[0], -config.data.output_size:])
        test_iter = config.test.n_iter

        logging.info("Starting training loop at epoch %d." % (initial_step,))
        scores_max = np.inf

        likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)

        for epoch in range(initial_step, config.training.epoch):
        # for epoch in range(initial_step, 10):


            state['epoch'] += 1
            for iteration, batch in enumerate(train_iter):
                batch = batch.to(config.device).float()
                loss = train_step_fn(state, batch)
                writer.add_scalar("training_loss", loss.item(), state['step'])

            if state['epoch'] % 50 == 0:
                save_checkpoint(checkpoint_meta_dir, state)  # save the model each epoch
            logging.info("epoch: %d, training_loss: %.5e" % (epoch, loss.item()))

            if epoch%10 ==0 :
                # ema.store(score_model.parameters())
                # ema.copy_to(score_model.parameters())
                with torch.no_grad():
                    sample, n = sampling_fn(score_model, sampling_shape=sampling_shape)
                #sample = sample[~torch.isnan(sample).any(dim=1)]
                sample = apply_activate(sample, transformer.output_info)
                # ema.restore(score_model.parameters())

                # train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
                #                                                     uniform_dequantization=True, evaluation=True)
                # inverse_transform function convert categorical in to order

                sample = torch.tensor(sample).float().to(config.device)
                if config.training.cond_y:
                    dis = calculate_wasserstein(sample,train_ds_[:sampling_shape[0],:-config.data.output_size],config)
                else:
                    dis = calculate_wasserstein(sample, train_ds_[:sampling_shape[0]], config)

                logging.info(f"epoch: {epoch}, wasserstein distance: {dis}")
                writer.add_scalar('wasserstein distance', torch.tensor(dis), epoch)

                if scores_max > torch.tensor(dis):
                    scores_max = torch.tensor(dis)
                    save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_max{ckp_fdix}.pth'), state)
            if epoch%500 ==0  and config.training.label and epoch> int(config.training.epoch*0.5) and False:
                cleaner_data = clean_fn(workdir.split('_')[0],config,train_ds,ckp_fdix)
                train_iter = DataLoader(cleaner_data, batch_size=config.training.batch_size, pin_memory=True)





        _, eval_ds_bpd, _ = datasets.get_dataset(config, uniform_dequantization=True, evaluation=True)
        ckpt_filename = os.path.join(checkpoint_dir, f"checkpoint_max{ckp_fdix}.pth")
        state = restore_checkpoint(ckpt_filename, state, device=config.device)
        logging.info(f"checkpoint : {state['step']}")
        ema.copy_to(state['model'].parameters())
        eval_samples = torch.tensor(eval_ds_bpd).float().to(config.device)

        sample_dis = 0
        sampling_shape = (1000 if len(eval_samples)>1000 else len(eval_samples), config.data.image_size)
        if cond_y:
            score_model = partial_model(state['model'], eval_samples[:sampling_shape[0], -config.data.output_size:])

        num_sampling_rounds = 5
        # # for 比赛：
        #
        # num_per_class = 1000
        # num_classes = config.data.output_size
        # sampling_shape = (num_per_class*num_classes, config.data.image_size)
        # # 生成 0-4 每个数各1000个的标签张量，形状为 (5000,)
        # labels = torch.arange(num_classes).repeat_interleave(num_per_class)
        # import torch.nn.functional as F
        # # 转换为 one-hot 编码，得到形状为 (5000, 5) 的整数张量
        # one_hot_int = F.one_hot(labels, num_classes=num_classes)
        #
        # # 转换为浮点型
        # one_hot_float = one_hot_int.float().to(config.device)
        # if cond_y:
        #     score_model = partial_model(state['model'], one_hot_float)
        #
        # ### finish
        for r in range(num_sampling_rounds):
            #ema.store(score_model.parameters())
            #ema.copy_to(score_model.parameters())
            with torch.no_grad():
                sample, n = sampling_fn(score_model, sampling_shape=sampling_shape)
            #ema.restore(score_model.parameters())
            sample = apply_activate(sample, transformer.output_info)
            sample = torch.tensor(sample).float().to(config.device)
            if config.training.cond_y:
                dis = calculate_wasserstein(sample[:2000], eval_samples[:2000][:sampling_shape[0], :-config.data.output_size], config)
            else:
                dis = calculate_wasserstein(sample[:2000], eval_samples[:2000][:sampling_shape[0]], config)
            sample_dis += dis
            # # for 比赛
            # np_array = sample.detach().cpu().numpy()
            #
            # # 利用 Pandas 保存为 CSV 文件（不保存索引）
            # df = pd.DataFrame(np_array)
            # df.to_csv(f'one_hot_tensor{ckp_fdix}-{r}.csv', index=False)
            # ###
        sample_dis /= num_sampling_rounds



        logging.info(f"average wasserstein distance {sample_dis}")
        if config.training.cond_y:
            self_dis = calculate_wasserstein(eval_samples[:1000,:-config.data.output_size],train_ds_[:1000,:-config.data.output_size],config)
        else:
            self_dis = calculate_wasserstein(eval_samples[:1000], train_ds_[:1000], config)

        logging.info(f"train data with test average wasserstein distance {self_dis}")

        logging.info('finish the train')
