import os

from sklearn.neural_network import MLPClassifier
from xgboost.dask import predict

from guidance_score import get_s_c,get_socre
import torch
import copy
import pandas as pd
from sklearn.model_selection import KFold
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
from models.utils import add_label_noise
from sklearn.metrics import accuracy_score, recall_score,precision_score,f1_score,roc_curve
from sklearn.cluster import OPTICS
from utils import apply_activate
import sde_lib
import datasets
import sampling
import numpy as np
import logging
from models import utils as mutils
from train_classify import load_labels
import random
import evaluation
import gc


FLAGS = flags.FLAGS
# 如果使用pycharm运行时，需要在路径前加../
flags.DEFINE_string("name", 'german',
                    "Training configuration.")
flags.DEFINE_string("order", '',
                    "Training configuration.")
flags.DEFINE_integer("timestep", 60, "sample timwstep range")
flags.DEFINE_integer("lam", 1700, "lambda --control the effect of classify ")
flags.DEFINE_float("thr", 0.2, "thrshold --control the thrshold of noise detect")
flags.DEFINE_float("noise", 0.05, "noise ratio --control the ratio of noise content")
flags.DEFINE_bool("cond_y", False, "if train score(x|y)")
flags.DEFINE_bool("xscore_label", False, "if train score(x, y)")
flags.mark_flags_as_required(["name", "order",  "lam", "thr", "noise", "cond_y", "xscore_label"])



def CWD(generator):
    # default give the noise rate
    cal_metric = lambda l1, p1: (
        accuracy_score(l1, p1), precision_score(l1, p1),
        recall_score(l1, p1), f1_score(l1, p1))
    downstream = [None]  # the None is a place for name of classify
    for train_ds, t_labels_num, eval_ds, eval_label, config_wy, train_ds_, eval_ds_, meta, noise_index, ckp_fdix, name in generator:
        logging.info(f"fold {ckp_fdix}")
        if meta['problem_type'] =='binary_classification':
            from contrast.Binary_clsss_CWD.CWD_binary import run as CWD_run
            from contrast.Binary_clsss_CWD.CWD_binary import evaluate as CWD_evaluate
            model = CWD_run(train_ds, t_labels_num, config_wy.training.noise)
            predict_out = model(train_ds).cpu().detach().view(-1)
            index = predict_out > 0.5
            predict_out[index] = 1.
            predict_out[~index] = 0.
            down_acc,down_precision,down_recall,down_f1 = CWD_evaluate(eval_ds, eval_label, model)
            predict = np.zeros(train_ds.shape[0])
            # 找到 a 和 b 不相等的位置，并将 c 中相应的位置设为 1
            predict[(predict_out != t_labels_num.detach().cpu()).numpy()] = 1

        else:
            from contrast.Multi_class_CWD.CWD import run as CWD_run
            from contrast.Multi_class_CWD.CWD import evaluate as CWD_evaluate
            model = CWD_run(train_ds,t_labels_num.long(),eval_ds,eval_label.long(),config_wy,train_ds_,eval_ds_,meta)
            predict_out = model(train_ds).cpu().detach()
            predict_out = torch.argmax(predict_out,dim=1)
            down_acc,down_precision,down_recall,down_f1 = CWD_evaluate(eval_ds, eval_label, model)
            predict = np.zeros(train_ds.shape[0])
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
    downstream = pd.concat(downstream, axis=1)

    downstream = pd.concat([pd.DataFrame([{'data_set': name}]), downstream],
                           axis=1)  # add the data name in the first row
    max_result = downstream
    return max_result

        

def ENN(generator,k=4):
    # default k==3
    from sklearn.neighbors import NearestNeighbors
    from collections import Counter

    # 定义函数来应用 Edited Nearest Neighbours
    K_set=[3,4,5,6,7,8,9,10,11,12,13,14,15]
    max_acc = 0
    max_score=None
    max_prect = None

    max_result = None
    cal_metric = lambda l1, p1: (
        accuracy_score(l1, p1), precision_score(l1, p1),
        recall_score(l1, p1), f1_score(l1, p1))

    for k in K_set:
        temp_acc_max = []
        downstream = [None]  # the None is a place for name of classify
        for train_ds, t_labels_num, eval_ds, eval_label, config_wy, train_ds_, eval_ds_, meta, noise_index, ckp_fdix, name in generator:
            logging.info(f"fold {ckp_fdix}")

            train_ds = train_ds.detach().cpu()
            t_labels_num = t_labels_num.detach().cpu()
            nbrs = NearestNeighbors(n_neighbors=k).fit(train_ds)
            indices = nbrs.kneighbors(train_ds, return_distance=False)[:,1:]  # remove index of itself, neighbors=3 when k=4 and after move itself
            predict = np.zeros(train_ds.shape[0])

            for i in range(len(train_ds)):
                # 获取最近邻标签
                neighbor_labels = t_labels_num[indices[i]]
                most_common_label = Counter(neighbor_labels).most_common(1)[0][0]

                # 只有当邻居中最常见的标签与当前标签一致时，才保留
                if most_common_label != t_labels_num[i]:
                    predict[i] = 1

            if sum(predict) == train_ds.shape[0]:
                down_acc, down_precision, down_recall, down_f1 = 0, 0, 0, 0

            else:
                scores, _ = evaluation.compute_scores(train_ds_, eval_ds_, [train_ds_[~predict.astype(bool)]],
                                                      metadata=meta)
                # this fold's result
                # to record this result
                predict, down_acc, down_precision, down_recall, down_f1 = predict, scores.iloc[:, 1:].values[0][
                    0], \
                    scores.iloc[:, 1:].values[0][1], scores.iloc[:, 1:].values[0][2], \
                    scores.iloc[:, 1:].values[0][3]
            performance = down_acc
            temp_acc_max.append(performance)
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

        if sum(temp_acc_max) >= max_acc:
            logging.info(f"best k is {k}")
            max_acc = sum(temp_acc_max)
            downstream = pd.concat(downstream, axis=1)

            downstream = pd.concat([pd.DataFrame([{'data_set': name}]), downstream],
                                   axis=1)  # add the data name in the first row
            max_result = downstream
    return max_result


def SR(g):
    # give hte tua,p and search the lamb
    from contrast.SparseRegularization import run
    return run(g)

def GCE(g):
    from contrast.GeneralizedCrossEntropy import run
    return run(g)

def INFFC(train_ds,t_labels_num,eval_ds,eval_label,config_wy,train_ds_,eval_ds_,meta):
    from contrast.INFFC import run
    predict = run(train_ds,t_labels_num,eval_ds,eval_label,config_wy,train_ds_,eval_ds_,meta)
    scores, _ = evaluation.compute_scores(train_ds_, eval_ds_, [train_ds_[~predict.astype(bool)]], metadata = meta)
    return predict, scores.iloc[:, 1:].values[0][0]

def mCRF(generator):
    cal_metric = lambda l1, p1: (
        accuracy_score(l1, p1), precision_score(l1, p1),
        recall_score(l1, p1), f1_score(l1, p1))
    from contrast.mCRF import crfnfl_all

    downstream = [None]  # the None is a place for name of classify
    for train_ds, t_labels_num, eval_ds, eval_label, config_wy, train_ds_, eval_ds_, meta, noise_index, ckp_fdix, name in generator:
        logging.info(f"fold {ckp_fdix}")
        if 1 or meta['problem_type'] =='binary_classification':
            classifier = MLPClassifier(
            hidden_layer_sizes=(int((train_ds.shape[0] + 1) * 2 / 3),),
            solver='adam',
            learning_rate_init=0.001,
            max_iter=200,
            activation='relu'
             )
        train_ds = train_ds.detach().cpu()
        t_labels_num = t_labels_num.detach().cpu()
        # the label is the first column
        train = np.concatenate((t_labels_num.reshape(-1, 1),train_ds), axis=1)
        # dd is the VCV, train_kmeans is the AM, predict is the new item
        dd, save_best = crfnfl_all(classifier, train)
        predict = np.ones_like(t_labels_num,dtype=bool)
        predict[dd] = False
        if len(dd) == 0:
            predict = np.ones_like(t_labels_num)
            down_acc, down_precision, down_recall, down_f1 = 0, 0, 0, 0
        else:
            scores, _ = evaluation.compute_scores(train_ds_, eval_ds_, [train_ds_[~predict]],
                                                  metadata=meta)
            # to record this result
            predict, down_acc, down_precision, down_recall, down_f1 = predict, scores.iloc[:, 1:].values[0][0], \
                scores.iloc[:, 1:].values[0][1], scores.iloc[:, 1:].values[0][2], \
                scores.iloc[:, 1:].values[0][3]

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
    downstream = pd.concat(downstream, axis=1)

    downstream = pd.concat([pd.DataFrame([{'data_set': name}]), downstream],
                           axis=1)  # add the data name in the first row
    max_result = downstream
    return max_result


def mDF(generator,k=3):
    from contrast.MuticalssRF import  return_predict
    import pandas as pd


    K_set=[3,4,5,6,7,8,9,10,11,12,13,14,15]
    max_acc = 0

    max_prect = None
    max_result = None
    cal_metric = lambda l1, p1: (
        accuracy_score(l1, p1), precision_score(l1, p1),
        recall_score(l1, p1), f1_score(l1, p1))


    for k in K_set:
        temp_acc_max = []
        downstream = [None]  # the None is a place for name of classify
        for train_ds, t_labels_num, eval_ds, eval_label, config_wy, train_ds_, eval_ds_, meta, noise_index, ckp_fdix, name in generator:
            logging.info(f"fold {ckp_fdix}")
            train_ds = train_ds.detach().cpu()
            t_labels_num = t_labels_num.detach().cpu()
            train = np.concatenate(( t_labels_num.reshape(-1, 1),train_ds), axis=1)
            df = pd.DataFrame(train)
            df = df.reset_index(drop=True)
            predict = return_predict(df, k, p=1)
            if sum(predict) == train_ds.shape[0]:
                down_acc, down_precision, down_recall, down_f1 = 0,0,0,0

            else:
                scores, _ = evaluation.compute_scores(train_ds_, eval_ds_, [train_ds_[~predict.astype(bool)]], metadata=meta)
                # this fold's result
                # to record this result
                predict, down_acc, down_precision, down_recall, down_f1 = predict, scores.iloc[:, 1:].values[0][0], \
                scores.iloc[:, 1:].values[0][1], scores.iloc[:, 1:].values[0][2], \
                    scores.iloc[:, 1:].values[0][3]
            performance = down_acc
            temp_acc_max.append(performance)
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

        if sum(temp_acc_max) >= max_acc:
            max_acc = sum(temp_acc_max)
            downstream = pd.concat(downstream, axis=1)

            downstream = pd.concat([pd.DataFrame([{'data_set': name}]), downstream],
                                   axis=1)  # add the data name in the first row
            max_result = downstream
    return max_result


def f_div(g):
    from contrast.fdivergence import run
    return run(g)

def SCE(g):
    from contrast.SymmetricCE import run
    return run(g)

def Nan(generator):
    cal_metric = lambda l1, p1: (
        accuracy_score(l1, p1), precision_score(l1, p1),
        recall_score(l1, p1), f1_score(l1, p1))
    downstream = [None]  # the None is a place for name of classify
    for train_ds, t_labels_num, eval_ds, eval_label, config_wy, train_ds_, eval_ds_, meta, noise_index, ckp_fdix, name in generator:
        logging.info(f"fold {ckp_fdix}")
        predict = np.zeros(train_ds.shape[0])
        scores, _ = evaluation.compute_scores(train_ds_, eval_ds_, [train_ds_[~predict.astype(bool)]], metadata = meta)
        predict, down_acc, down_precision, down_recall, down_f1 = predict, scores.iloc[:, 1:].values[0][0], scores.iloc[:, 1:].values[0][1], scores.iloc[:, 1:].values[0][2], \
    scores.iloc[:, 1:].values[0][3]

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

    downstream = pd.concat(downstream, axis=1)

    downstream = pd.concat([pd.DataFrame([{'data_set': name}]), downstream],
                           axis=1)  # add the data name in the first row
    max_result = downstream
    return max_result


def GMM_CT(g):
    from contrast.GMM_CT.dynamic_sample_selection import GMM_CT_run

    return GMM_CT_run(g)

def LW(g):
    from contrast.Label_Wave import run
    return run(g)

_MODEL = {
    # 'CWD':CWD,
    # 'SR':SR,
    # 'f_div':f_div,
    # 'SCE': SCE,
    # 'GCE':GCE,
    # 'ENN':ENN,
    #   # 'INFFC':INFFC,  # bug
     'mCRF':mCRF,
    # 'Nan':Nan,
    # 'GMM_CT': GMM_CT,
    # 'LW':LW,
    #  'mDF':mDF

}

def get_the_data(kf,data_name,data,transformer,config_wy,meta):
    ckp_fdix = 0
    result = []

    for train_index, val_index in kf.split(data):
        randomSeed = 2021
        torch.manual_seed(randomSeed)
        torch.cuda.manual_seed(randomSeed)
        torch.cuda.manual_seed_all(randomSeed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(randomSeed)
        random.seed(randomSeed)
        gc.collect()

        ckp_fdix += 1
        train_ds, eval_ds = data[train_index], data[val_index]
        eval_ds_ = transformer.inverse_transform(eval_ds)  # for downstream task
        eval_ds = torch.tensor(eval_ds).to(config_wy.device).float()

        np.random.seed(randomSeed)
        real_labels = train_ds[:, -config_wy.data.output_size:]
        train_ds = train_ds[:, :-config_wy.data.output_size]

        t_labels = real_labels.copy()

        # cluster_index = train_ds[:,-1]
        # filtered_rows = train_ds[cluster_index < 1]
        # cluster(filtered_rows)

        # add noise for label
        t_labels, noise_index = add_label_noise(t_labels, config_wy)

        '''
        # for synthesize data
        l = 200
        rnnn = 10
        for i in range(l-rnnn,l):noise_index[i]= 1
        for i in range(2*l-rnnn,2*l):noise_index[i]= 1
        '''
        train_ds = torch.tensor(train_ds).to(config_wy.device).float()
        t_labels_oh = torch.tensor(t_labels).to(config_wy.device).float()  # onehot
        t_labels_num = torch.argmax(t_labels_oh, dim=1).float()  # num type
        eval_label = torch.argmax(eval_ds[:, -config_wy.data.output_size:], dim=1).to(config_wy.device).float()

        train_ds_ = transformer.inverse_transform(torch.cat((train_ds, t_labels_oh), dim=1).detach().cpu().numpy())
        # point to evaluate the other method,need a bool list of noise predict and a acc
        # different way for robust method and filter method

        # the input train set and eval set is feature without label, but the train_ is the data with label

        result.append((train_ds, t_labels_num, eval_ds[:,:-config_wy.data.output_size],eval_label, config_wy, train_ds_,eval_ds_, meta,noise_index,ckp_fdix,data_name))
    return result

def main(avg):
    name = FLAGS.name


    config_flags.DEFINE_config_file("config_condy", 'configs/{}.py'.format(name), "Training configuration.",
                                    lock_config=True)

    flags.DEFINE_string("path_woy", '{}_woy_xscore/checkpoints/checkpoint_max.pth'.format(name)
                        , "Training configuration.")
    flags.DEFINE_string("path_wy", '{}_condy_xscore/checkpoints/checkpoint_max.pth'.format(name),
                        "Training configuration.")
    flags.DEFINE_string("path_classify", '{}_classify/checkpoints/checkpoint_max.pth'.format(name),
                        "Training configuration.")

    FLAGS.config_condy.training.noise = FLAGS.noise
    config_wy = copy.deepcopy(FLAGS.config_condy)

    config_woy = copy.deepcopy(config_wy)

    # for read data with y, if this is false will read data without y
    FLAGS.xscore_label = False  # if train the combine distribution p(x,y)
    FLAGS.cond_y = True  # if train the feature condition with y, set xscore_label==False
    mutils.after_defined(FLAGS, config_wy)

    FLAGS.xscore_label = False  # if train the combine distribution p(x,y)
    FLAGS.cond_y = False  # if train the feature condition with y, set xscore_label==False
    mutils.after_defined(FLAGS, config_woy)

    path_woy = FLAGS.path_woy
    path_wy = FLAGS.path_wy
    path_classify = FLAGS.path_classify

    cal_metric = lambda l1, p1: (
        accuracy_score(l1, p1), precision_score(l1, p1),
        recall_score(l1, p1), f1_score(l1, p1))

    #
    # model(x, t) t<1 ,t~[0.01,10]
    train_ds, eval_ds, (transformer, meta) = datasets.get_dataset(config_wy,
                                                                  uniform_dequantization=config_wy.data.uniform_dequantization)
    if meta['problem_type'] == 'binary_classification':
        metric = 'binary_f1'
    elif meta['problem_type'] == 'regression':
        metric = "r2"
    else:
        metric = 'macro_f1'
    data = np.concatenate((train_ds, eval_ds), axis=0)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for model_name in _MODEL.keys():
        # set randomSeed

        logging.info(f"model name  {model_name}")
        ckp_fdix = 0

        # point to evaluate the other method,need a bool list of noise predict and a acc
        # different way for robust method and filter method

        # the input train set and eval set is feature without label, but the train_ is the data with label
        data_generater = get_the_data(kf,name,data,transformer,config_wy,meta)
        downstream = _MODEL[model_name](data_generater)

        os.makedirs('./contrast_result', exist_ok=True)
        file_path = f'./contrast_result/{config_wy.training.noise}_results_{model_name}.csv'

        # 如果文件不存在，写入数据并包含列索引名称
        if not pd.io.common.file_exists(file_path):
            downstream.to_csv(file_path, index=False,columns=downstream.columns)
        else:
            # 如果文件已存在，则追加数据，并且不写入列索引名称
            downstream.to_csv(file_path, mode='a', header=False, index=False,columns=downstream.columns)

if __name__ == "__main__":
    app.run(main)

