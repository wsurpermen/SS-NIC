# Import libraries
import logging

import numpy as np
import os
import csv
import math

import pandas as pd
import torch.nn.parallel
from torch.utils.data import DataLoader, Dataset
from contrast.model import MLP
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import torch.optim
import torch.utils.data
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score

torch.autograd.set_detect_anomaly(True)
num_epochs = 1000
CUDA = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

CE = nn.CrossEntropyLoss().cuda()



# Stable version of CE Loss
class CrossEntropyLossStable(nn.Module):
    def __init__(self, reduction='mean', eps=1e-5):
        super(CrossEntropyLossStable, self).__init__()
        self._name = "Stable Cross Entropy Loss"
        self._eps = eps
        self._softmax = nn.Softmax(dim=-1)
        self._nllloss = nn.NLLLoss(reduction=reduction)

    def forward(self, outputs, labels):
        return self._nllloss(torch.log(self._softmax(outputs) + self._eps), labels)


criterion = CrossEntropyLossStable()
criterion.cuda()

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return idx, self.features[idx], self.labels[idx]

def return_opt(div):
    global activation
    global conjugate
    if div == 'KL':
        def activation(x):
            return -torch.mean(x)

        def conjugate(x):
            return -torch.mean(torch.exp(x - 1.))

    elif div == 'Reverse-KL':
        def activation(x):
            return -torch.mean(-torch.exp(x))

        def conjugate(x):
            return -torch.mean(-1. - x)  # remove log

    elif div == 'Jeffrey':
        def activation(x):
            return -torch.mean(x)

        def conjugate(x):
            return -torch.mean(x + torch.mul(x, x) / 4. + torch.mul(torch.mul(x, x), x) / 16.)

    elif div == 'Squared-Hellinger':
        def activation(x):
            return -torch.mean(1. - torch.exp(x))

        def conjugate(x):
            return -torch.mean((1. - torch.exp(x)) / (torch.exp(x)))

    elif div == 'Pearson':
        def activation(x):
            return -torch.mean(x)

        def conjugate(x):
            return -torch.mean(torch.mul(x, x) / 4. + x)

    elif div == 'Neyman':
        def activation(x):
            return -torch.mean(1. - torch.exp(x))

        def conjugate(x):
            return -torch.mean(2. - 2. * torch.sqrt(1. - x))

    elif div == 'Jenson-Shannon':
        def activation(x):
            return -torch.mean(- torch.log(1. + torch.exp(-x))) - torch.log(torch.tensor(2.))

        def conjugate(x):
            return -torch.mean(x + torch.log(1. + torch.exp(-x))) + torch.log(torch.tensor(2.))

    elif div == 'Total-Variation':
        def activation(x):
            return -torch.mean(torch.tanh(x) / 2.)

        def conjugate(x):
            return -torch.mean(torch.tanh(x) / 2.)

    else:
        raise NotImplementedError("[-] Not Implemented f-divergence %s" % div)


def run(generator,opt='Total-Variation'):

    max_acc = 0
    max_f = -100
    params = None
    cal_metric = lambda l1, p1: (
        accuracy_score(l1, p1), precision_score(l1, p1),
        recall_score(l1, p1), f1_score(l1, p1))
    downstream = [None]  # the None is a place for name of classify
    for train_ds, t_labels_num, eval_ds, eval_label, config_wy, train_ds_, eval_ds_, meta, noise_index, ckp_fdix, name in generator:
        logging.info(f"fold {ckp_fdix}")
        # Divergence functions:
        return_opt(opt)
        origin_ds = train_ds
        origin_label = t_labels_num.long()
        model_prob = MLP(config_wy.data.image_size, config_wy.data.output_size).to(config_wy.device)
        best_prob_acc = 0
        max_f = -100
        val_acc_noisy_result = []
        train_acc_result = []
        test_acc_result = []
        f_result = []
        f_test_result = []
        num_train = int(train_ds.shape[0] * 0.8)
        train_ds, val_ds = train_ds[:num_train], train_ds[num_train:]
        t_labels_num, v_labels_num = t_labels_num[:num_train], t_labels_num[num_train:]

        # Dataloader for peer samples, which is used for the estimation of the marginal distribution
        peer_train = torch.utils.data.DataLoader(
            CustomDataset(train_ds, t_labels_num),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True)
        train_loader_noisy = torch.utils.data.DataLoader(
            CustomDataset(train_ds, t_labels_num),
            batch_size=batch_size,
            shuffle=True
        )
        peer_val = torch.utils.data.DataLoader(
            CustomDataset(val_ds, v_labels_num),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True)
        valid_loader_noisy = torch.utils.data.DataLoader(
            CustomDataset(val_ds, v_labels_num),
            batch_size=batch_size,
            shuffle=True
        )

        # Below we provide two learning rate settings which are used for all experiments in MNIST
        for epoch in range(num_epochs):
            model_prob.train()
            # print("epoch=", epoch)
            # Setting 1
            learning_rate = 5e-3
            if epoch > 200:
                learning_rate = 2e-3
            elif epoch > 400:
                learning_rate = 1e-3
            elif epoch > 600:
                learning_rate = 1e-4
            # elif epoch > 80:
            #     learning_rate = 1e-5

            # Setting 2
            #        learning_rate = 5e-4
            #        if epoch > 20:
            #            learning_rate = 1e-4
            #        elif epoch > 40:
            #            learning_rate = 5e-4
            #        elif epoch > 60:
            #            learning_rate = 1e-5
            #        elif epoch > 80:
            #            learning_rate = 1e-5

            # We adopted the ADAM optimizer
            optimizer_prob = torch.optim.Adam(model_prob.parameters(), lr=learning_rate)
            train(train_loader=train_loader_noisy, peer_loader=peer_train, model=model_prob, optimizer=optimizer_prob,
                  epoch=epoch)
            # train_acc = evaluate(train_ds,t_labels_num,model_prob)
            f_div_value = f_calculate(model_prob, valid_loader_noisy, peer_val)
            if f_div_value > max_f:
                max_f = f_div_value
                params = model_prob.state_dict()

        model = MLP(config_wy.data.image_size, config_wy.data.output_size).to(config_wy.device)
        model.load_state_dict(params)

        down_acc, down_precision, down_recall, down_f1 = evaluate(eval_ds, eval_label, model, meta)
        predict = np.zeros(origin_ds.shape[0])
        predict_out = model(origin_ds).cpu().detach()
        predict_out = torch.argmax(predict_out, dim=1)
        # 找到 a 和 b 不相等的位置，并将 c 中相应的位置设为 1
        predict[(predict_out != origin_label.detach().cpu()).numpy()] = 1

        test_labels = np.zeros(origin_ds.shape[0])
        n_s = [i for i in range(len(noise_index)) if noise_index[i]]  # noise sample's index
        test_labels[n_s] = 1.
        acc, precision, recall, f1 = cal_metric(test_labels, predict)

        suc_score = pd.DataFrame([{
            f'{ckp_fdix}noise_count': sum(noise_index),
            f'{ckp_fdix}predict_count': sum(predict),
            f'{ckp_fdix}acc': acc,
            f'{ckp_fdix}precise': precision,
            f'{ckp_fdix}recall': recall,
            f'{ckp_fdix}down_acc': down_acc,
            f'{ckp_fdix}down_precision': down_precision,
            f'{ckp_fdix}down_recall': down_recall,
            f'{ckp_fdix}down_f1': down_f1
        }])
        downstream.append(suc_score)

    downstream = pd.concat(downstream, axis=1)

    downstream = pd.concat([pd.DataFrame([{'data_set': name}]), downstream],
                           axis=1)  # add the data name in the first row
    max_result = downstream
    return max_result

def evaluate(test_attr,label,model,meta=None):
    model.eval()
    predict_out = model(test_attr).cpu().detach().numpy()
    predict_out = np.argmax(predict_out, axis=1)
    if meta==None: # for choice best model
        return accuracy_score(label.data.cpu().numpy(), predict_out)
    if meta['problem_type'] == 'binary_classification':
        return accuracy_score(label.data.cpu().numpy(), predict_out), precision_score(label.data.cpu().numpy(),
                                                                              predict_out), recall_score(
            label.data.cpu().numpy(), predict_out), f1_score(label.data.cpu().numpy(), predict_out, average='binary')
    else:
        return accuracy_score(label.data.cpu().numpy(), predict_out), precision_score(label.data.cpu().numpy(),
                                                                              predict_out,
                                                                              average='macro'), recall_score(
            label.data.cpu().numpy(), predict_out, average='macro'), f1_score(label.data.cpu().numpy(), predict_out, average='macro')

# Stable PROB: returns the negative predicted probability of an image given a reference label
class ProbLossStable(nn.Module):
    def __init__(self, reduction='none', eps=1e-5):
        super(ProbLossStable, self).__init__()
        self._name = "Prob Loss"
        self._eps = eps
        self._softmax = nn.Softmax(dim=-1)
        self._nllloss = nn.NLLLoss(reduction='none')

    def forward(self, outputs, labels):
        return self._nllloss(self._softmax(outputs), labels)


criterion_prob = ProbLossStable()
criterion_prob.cuda()

batch_size = 1000


def train(train_loader, peer_loader, model, optimizer, epoch, warmup=0):
    model.train()
    for i, (idx, input, target) in enumerate(train_loader):
        if idx.size(0) != batch_size:
            continue
        warmup_epoch = warmup
        input = torch.autograd.Variable(input.cuda())
        target = torch.autograd.Variable(target.long().cuda())
        output = model(input)
        optimizer.zero_grad()
        # After warm-up epochs, switch to optimizing f-divergence functions
        if epoch >= warmup_epoch:

            # Estimate E_Z [g(Z)] where Z follows the joint distribution of h, noisy Y;
            # g is the activation function
            prob_reg = -criterion_prob(output, target)
            loss_regular = activation(prob_reg)

            # Estimate E_Z [f^*(g(Z))] where Z follows the product of marginal distributions of h, noisy Y;
            # f^*(g) is the conjugate function;
            peer_iter = iter(peer_loader)
            try:
                input1 = next(peer_iter)[1]
                target2 = next(peer_iter)[2]
            except StopIteration:
                break
            input1 = torch.autograd.Variable(input1.cuda())
            output1 = model(input1)
            target2 = torch.autograd.Variable(target2.long().cuda())
            prob_peer = -criterion_prob(output1, target2)
            loss_peer = conjugate(prob_peer)
            loss = loss_regular - loss_peer
        # Use CE loss for the warm-up.
        else:
            loss = criterion(output, target)
        loss.cuda()
        loss.backward()
        optimizer.step()

# Calculate f-divergence value in the max game
def f_calculate(model, data_loader, peer_loader):
    model.eval()
    f_score = 0
    for i, (idx, input, target) in enumerate(data_loader):
        if idx.size(0) != batch_size:
            continue
        input = torch.autograd.Variable(input.cuda())
        target = torch.autograd.Variable(target.long().cuda())
        output = model(input)
        prob_reg = -criterion_prob(output.detach(), target)
        loss_regular = activation(prob_reg)
        peer_iter = iter(peer_loader)
        try:
            input1 = next(peer_iter)[1]
            target2 = next(peer_iter)[2]
        except StopIteration:
            break
        input1 = torch.autograd.Variable(input1.cuda())
        output1 = model(input1)
        target2 = torch.autograd.Variable(target2.long().cuda())
        prob_peer = -criterion_prob(output1.detach(), target2)
        loss_peer = conjugate(prob_peer)
        score = loss_peer - loss_regular
        f_score += score * target.size(0)
    return f_score/10000
