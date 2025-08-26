import logging

import pandas as pd
import torch
import numpy as np
from contrast.model import MLP
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score

class pNorm(nn.Module):
    def __init__(self, p=0.5):
        super(pNorm, self).__init__()
        self.p = p

    def forward(self, pred, p=None):
        if p:
            self.p = p
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1)
        norm = torch.sum(pred ** self.p, dim=1)
        return norm.mean()

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
                predict_out, average='macro'), recall_score(
            label.data.cpu().numpy(), predict_out, average='macro'), f1_score(label.data.cpu().numpy(), predict_out, average='macro')


def run(generator):
    tau, p= 0.5, 0.1,
    lamb_set = {0.1, 0.3, 1.0, 3.0, 5.0, 7.0, 10, 15, 20}
    lr = 0.001

    params =  None
    max_acc = 0
    epoch = 200
    max_result = None
    cal_metric = lambda l1, p1: (
        accuracy_score(l1, p1), precision_score(l1, p1),
        recall_score(l1, p1), f1_score(l1, p1))

    for lamb in lamb_set:
        temp_acc_max = []
        downstream = [None]  # the None is a place for name of classify
        for train_ds, t_labels_num, eval_ds, eval_label, config_wy, train_ds_, eval_ds_, meta,noise_index,ckp_fdix,name in generator:
            logging.info(f"fold {ckp_fdix}")
            norm = pNorm(p)
            model = MLP(config_wy.data.image_size, config_wy.data.output_size).to(config_wy.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            model.train()
            for _ in range(epoch):
                optimizer.zero_grad()
                out = model(train_ds)
                loss = F.cross_entropy(out/tau, t_labels_num.long())+ lamb * norm(out / tau, p)
                loss.backward()
                optimizer.step()
            # this fold's result
            performance = evaluate(train_ds, t_labels_num, model)
            temp_acc_max.append(performance)
            # to record this result
            down_acc, down_precision, down_recall, down_f1 = evaluate(eval_ds, eval_label, model, meta)
            predict = np.zeros(train_ds.shape[0])
            predict_out = model(train_ds).cpu().detach()
            predict_out = torch.argmax(predict_out, dim=1)
            # 找到 a 和 b 不相等的位置，并将 c 中相应的位置设为 1
            predict[(predict_out != t_labels_num.detach().cpu()).numpy()] = 1

            test_labels = np.zeros(train_ds.shape[0])
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

        if sum(temp_acc_max) > max_acc:
            max_acc = sum(temp_acc_max)
            downstream = pd.concat(downstream, axis=1)

            downstream = pd.concat([pd.DataFrame([{'data_set': name}]), downstream],
                                   axis=1)  # add the data name in the first row
            max_result = downstream
    return max_result




