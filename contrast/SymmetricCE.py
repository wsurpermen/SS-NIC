import logging

import torch
import numpy as np
from contrast.model import MLP
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import torch.nn as nn
import itertools
import pandas as pd
eps = 1e-7
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score

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

class SCELoss(nn.Module):
    def __init__(self, num_classes=10, a=1, b=1):
        super(SCELoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.b = b
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        ce = self.cross_entropy(pred, labels)
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        loss = self.a * ce + self.b * rce.mean()
        return loss


def run(generator):
    lr = 0.002
    a_set = {0.01,0.1,1,10,20}
    b_set = {0.1,1}
    params =  None
    max_acc = 0
    epoch = 1000
    max_result = None
    cal_metric = lambda l1, p1: (
        accuracy_score(l1, p1), precision_score(l1, p1),
        recall_score(l1, p1), f1_score(l1, p1))

    for a,b in itertools.product(a_set, b_set):
        temp_acc_max = []
        downstream = [None]  # the None is a place for name of classify
        for train_ds, t_labels_num, eval_ds, eval_label, config_wy, train_ds_, eval_ds_, meta,noise_index,ckp_fdix,name in generator:
            logging.info(f"fold {ckp_fdix}")
            loss_fn = SCELoss(num_classes=config_wy.data.output_size,a=a,b=b)
            model = MLP(config_wy.data.image_size, config_wy.data.output_size).to(config_wy.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            model.train()
            for _ in range(epoch):
                model.train()
                optimizer.zero_grad()
                out = model(train_ds)
                loss = loss_fn.forward(out,t_labels_num.long())
                loss.backward()
                optimizer.step()

            # this fold's result
            performance = evaluate(train_ds,t_labels_num,model)
            temp_acc_max.append(performance)
            # to record this result
            down_acc,down_precision,down_recall,down_f1 = evaluate(eval_ds, eval_label, model, meta)
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

    # model = MLP(config_wy.data.image_size, config_wy.data.output_size).to(config_wy.device)
    # model.load_state_dict(params)
    #
    # predict_out = model(train_ds).cpu().detach()
    # predict_out = torch.argmax(predict_out, dim=1)
    # max_acc,down_precision,down_recall,down_f1 = evaluate(eval_ds,eval_label,model,meta)
    # predict = np.zeros(train_ds.shape[0])
    # # 找到 a 和 b 不相等的位置，并将 c 中相应的位置设为 1
    # predict[(predict_out != t_labels_num.detach().cpu()).numpy()] = 1
    # return predict, max_acc,down_precision,down_recall,down_f1