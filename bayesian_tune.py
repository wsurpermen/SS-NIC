from importlib.metadata import metadata

from numpy.ma.extras import mask_rows
from scipy.constants import precision
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode  # 用于找出投票最多的标签
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import torch.nn.functional as F
from guidance_score import get_socre, get_class
import torch
import copy
import pandas as pd
from sklearn.model_selection import KFold
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
from models.utils import add_label_noise
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve
from sklearn.cluster import OPTICS
from sklearn.neural_network import MLPClassifier

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
import optuna

'''
for condition distribution p(x|y)
'''

FLAGS = flags.FLAGS
flags.DEFINE_string("name", 'ESL',
                    "Training configuration.")
flags.DEFINE_string("order", '',
                    "Training configuration.")
flags.DEFINE_integer("timestep", 60, "sample timwstep range")
flags.DEFINE_integer("lam", 1700, "lambda --control the effect of classify ")
flags.DEFINE_float("thr", 0.2, "thrshold --control the thrshold of noise detect")
flags.DEFINE_float("noise", 0.05, "noise ratio --control the ratio of noise content")
flags.DEFINE_bool("cond_y", False, "if train score(x|y)")
flags.DEFINE_bool("xscore_label", False, "if train score(x, y)")
flags.mark_flags_as_required(["name", "order", "lam", "thr", "noise", "cond_y", "xscore_label"])

# set randomSeed, for global data
randomSeed = 2021
torch.manual_seed(randomSeed)
torch.cuda.manual_seed(randomSeed)
torch.cuda.manual_seed_all(randomSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(randomSeed)
random.seed(randomSeed)
kf_times = 10


def cosine_distance(vec1, vec2):
    """计算两个向量之间的余弦距离"""
    dot_product = np.sum(vec1 * vec2, axis=1)
    norm1 = np.linalg.norm(vec1, axis=1)
    norm2 = np.linalg.norm(vec2, axis=1)
    return 1 - dot_product / (norm1 * norm2)


def get_s_c(config, model_s, model_c, eval=False, sde=None, condition=False, lambda_=20):
    '''
    return guidance of diffusion score
    '''
    if eval:
        if isinstance(model_s, torch.nn.DataParallel):  # for DataParaller
            model_s = model_s.module
        if isinstance(model_c, torch.nn.DataParallel):  # for DataParaller
            model_c = model_c.module

    def func(x, t, y_cond=None, t2=None):
        if not t2:  # if part one and part two need different time step
            t2 = t
        if not condition:
            score = model_s(x, t)
        else:
            score = model_s(x, t, y_cond=y_cond)
        with torch.enable_grad():
            if config.training.sde == 'vesde':
                t2 = sde.marginal_prob(torch.zeros_like(x), t)[1]
            x_in = x.detach().requires_grad_(True)
            logits = model_c(x_in, t2)
            log_probs = F.log_softmax(logits, dim=-1)
            # notice the labels's shape, if it is a 2D tensor, the selected's shape will be 2D too
            selected = log_probs[range(len(logits)), torch.argmax(y_cond, dim=1).long().view(-1)]
            grad = torch.autograd.grad(selected.sum(), x_in)[0]
            # return torch.autograd.grad(selected.sum(), x_in)[0] * lambda_ + score
            # return (grad* lambda_ + score, score, grad)
            return grad * lambda_ + score

    return func


def make_threshold(tensor, factor=0.8):
    way = 0
    t = 0
    if way == 0:  # base on IQR
        a = torch.median(tensor) + (torch.quantile(tensor, 0.75) - torch.quantile(tensor, 0.25)) * factor
        for i in range(tensor.shape[0] - 1, 0, -1):
            if tensor[i] < a:
                t = i
                break

    elif way == 1:  # base on z-score
        mean = torch.mean(tensor)
        std = torch.std(tensor)

        # 计算 z-score
        z_scores = (tensor - mean) / std
        for i in range(z_scores.shape[0] - 1, 0, -1):
            if z_scores[i] < 1.5:
                t = i
                break
    return ((tensor.shape[0] - t) / tensor.shape[0])


def create_object(Model_list):
    def objective(trial):
        param = {
            'cosine_factor': trial.suggest_float('cosine_factor', 0, 10),  # 撞上下墙
            'filter_factor': trial.suggest_float('filter_factor', 0.3, 2.0),  # 撞上下墙
            'time_step': trial.suggest_int('time_step', 50, 200),
            'classify_lambda': trial.suggest_int('classify_lambda', 0, 70),  # 撞上下墙
            'k_nbrs': trial.suggest_int('k_nbrs', 5, 15),  # 撞上墙
            'easy_fliter': trial.suggest_float('easy_fliter', 0.1, 0.5),

            # 'filter_factor': config_wy.filter_factor,
            # 'time_step': config_wy.time_step,
            # 'classify_lambda': config_wy.classify_lambda,
            # 'k_nbrs':config_wy.k_nbrs,
            # 'easy_fliter':config_wy.easy_fliter
        }

        def integration_find(noise_data, p_i):
            # 初始化RandomForest模型，n_estimators是模型数量（树的数量）
            n_estimators = 100  # 可以调整为100-500之间
            predict = np.ones(noise_data.shape[0], dtype='bool')
            predict[p_i] = False
            cleaner_data = noise_data[predict]
            X_train = cleaner_data[:, :-config_wy.data.output_size]
            y_train = np.argmax(cleaner_data[:, -config_wy.data.output_size:], axis=1)
            unique_labels = np.unique(y_train)

            # 创建标签到新索引的映射
            label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

            # 创建反向映射，从新索引映射回原始标签
            reverse_label_mapping = {v: k for k, v in label_mapping.items()}

            # 用新的标签替换y_train中的标签
            y_train = np.vectorize(label_mapping.get)(y_train)

            # if len(unique_labels) == 1:
            #     one_hot_y = np.zeros(config_wy.data.output_size)
            #     one_hot_y[y_train[0]] = 1
            #     return np.array([one_hot_y for i in range(len(p_i))])
            un_data = noise_data[~predict]
            un_feature = un_data[:, :-config_wy.data.output_size]
            un_labels = un_data[:, -config_wy.data.output_size:]
            rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            # 训练模型
            rf_model.fit(X_train, y_train)
            # 初始化XGBoost模型
            xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)  # 100 是弱学习器数量
            # 训练模型
            xgb_model.fit(X_train, y_train)
            rf_predictions = rf_model.predict(un_feature)
            xgb_predictions = xgb_model.predict(un_feature)
            consistent_predictions = rf_predictions == xgb_predictions
            y_test_original = np.array([reverse_label_mapping[label] for label in xgb_predictions])
            one_hot_y = np.zeros((len(un_feature), config_wy.data.output_size))
            one_hot_y[np.arange(len(un_feature)), y_test_original] = 1

            un_data[consistent_predictions, -config_wy.data.output_size:] = one_hot_y[consistent_predictions]
            return np.concatenate((cleaner_data, un_data[consistent_predictions]), axis=0)

        def dis_fn(score_l_wy, score_l_woy, way=10, direct1=None, direct2=None, factor=2):
            cos_dis = torch.tensor(cosine_distance(score_l_wy, score_l_woy))
            l2_wy = torch.tensor(np.linalg.norm(score_l_wy, axis=-1))

            l2_woy = torch.tensor(np.linalg.norm(score_l_woy, axis=-1))
            ED = torch.tensor(np.linalg.norm(score_l_wy - score_l_woy, axis=-1))
            theta = torch.acos(1 - cos_dis) * (180 / torch.pi) + 10
            sin = torch.sin(theta)

            MD = torch.abs((l2_wy - l2_woy))
            MA = torch.abs((l2_wy + l2_woy))

            if way == 0:
                dis = MD * (1 + cos_dis)
            elif way == 1:
                dis = (MD - torch.min(MD)) / torch.max(MD) + cos_dis / 2
            elif way == 2:
                dis = cos_dis
            elif way == 3:
                dis = MD
            elif way == 4:
                dis = MD * (torch.where(cos_dis < 1, torch.tensor(1), torch.tensor(-1)))
            elif way == 5:
                dis = l2_wy * l2_woy * sin * theta * torch.pi * (ED + MD) ** 2 / 720
            elif way == 6:
                dis = MA * (1 + cos_dis)
            elif way == 7:
                dis = MA
            elif way == 8:
                cos_dis = torch.tensor(cosine_distance(direct1, direct2))
                dis = MA * (factor + cos_dis)
            elif way == 9:
                cos_dis = torch.tensor(cosine_distance(direct1, direct2))
                MA = torch.sum(MA, dim=0)
                dis = torch.unsqueeze((MA - torch.min(MA)) * 2 / (torch.max(MA) - torch.min(MA)) + factor * cos_dis, 0)
            elif way == 10:  # the discrepancy of un/su trajectory
                cos_dis = torch.tensor(cosine_distance(direct1, direct2))
                MD = torch.sum(MA, dim=0)
                dis = torch.unsqueeze((MD - torch.min(MD)) * 2 / (torch.max(MD) - torch.min(MD)) + factor * cos_dis, 0)

            dis_ = torch.sum(dis, dim=0)
            return dis_

        combine_class = False  # false to filter class by class
        del_correct = 2  # false to correct else del
        dis_way = 10

        # set randomSeed
        randomSeed = 2021
        torch.manual_seed(randomSeed)
        torch.cuda.manual_seed(randomSeed)
        torch.cuda.manual_seed_all(randomSeed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(randomSeed)
        random.seed(randomSeed)

        cal_metric = lambda l1, p1: (
            accuracy_score(l1, p1), precision_score(l1, p1),
            recall_score(l1, p1), f1_score(l1, p1))

        #

        kf = KFold(n_splits=kf_times, shuffle=True, random_state=42)
        ckp_fdix = 0
        avg_acc = 0
        avg_precision = 0
        avg_recall = 0
        avg_f1 = 0
        for train_index, val_index in kf.split(data):
            randomSeed = 2021
            torch.manual_seed(randomSeed)
            torch.cuda.manual_seed(randomSeed)
            torch.cuda.manual_seed_all(randomSeed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(randomSeed)
            random.seed(randomSeed)
            ckp_fdix += 1
            logging.info(f"fold {ckp_fdix}")
            train_ds, eval_ds = data[train_index], data[val_index]

            eval_ds_ = transformer.inverse_transform(eval_ds)  # for downstream task
            train_ds_ = transformer.inverse_transform(train_ds)

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
            t_labels = torch.tensor(t_labels).to(config_wy.device).float()

            # model_woy can be used as p(x|y) if input the sampling_fn label
            model_woy = Model_list[ckp_fdix]['woy']
            inverse_scaler_woy = datasets.get_data_inverse_scaler(config_woy)
            sampling_shape_woy = (eval_ds.shape[0], config_woy.data.image_size)
            sampling_fn_woy = sampling.get_sampling_fn(config_woy, sde, sampling_shape_woy, inverse_scaler_woy,
                                                       sampling_eps)

            # init model with lable y

            inverse_scaler_wy = datasets.get_data_inverse_scaler(config_wy)
            sampling_shape_wy = (eval_ds.shape[0], config_wy.data.image_size)
            sampling_fn_wy = sampling.get_sampling_fn(config_wy, sde, sampling_shape_wy, inverse_scaler_wy,
                                                      sampling_eps)

            # model: logp(x|y) + logp(y|x)'
            model_wy = get_s_c(config_wy, Model_list[ckp_fdix]['wy'], Model_list[ckp_fdix]['classify'], eval=True,
                               sde=sde, condition=True, lambda_=param['classify_lambda'])
            # model_wy = get_s_c(config_woy, path_woy, path_classify, eval= True, sde=sde, condition=False)
            # show_score(model_wy,model_woy)
            # return 0

            # if input mudel is p(x,y), need cut down the model's output

            # x_woy, score_l_woy = sampling_fn_woy(model_wy, sampling_shape=sampling_shape_wy, distance=True, semi_x=train_ds,
            #                                      labels=t_labels,
            #                                      timestep=param['time_step'], xscore_sample=True, xmodle=model_woy)
            #
            # x_wy, score_l_wy = sampling_fn_wy(model_wy, sampling_shape=sampling_shape_wy, distance=True, semi_x=train_ds,
            #                                   labels=t_labels,
            #                                   timestep=param['time_step'])

            x_woy, score_l_woy = sampling_fn_woy(model_woy, sampling_shape=sampling_shape_wy, distance=True,
                                                 semi_x=train_ds,
                                                 timestep=param['time_step'], condition_sample=False)

            x_wy, score_l_wy = sampling_fn_wy(model_wy, sampling_shape=sampling_shape_wy, distance=True,
                                              semi_x=train_ds,
                                              labels=t_labels,
                                              timestep=param['time_step'], condition_sample=True)

            # direct1 = (x_woy - train_ds).detach().cpu().numpy()# 计算数据原点距离两个质心之间的距离
            # direct2 = (x_wy - train_ds).detach().cpu().numpy()

            direct1 = (x_woy).detach().cpu().numpy()  # 计算最终两个收敛点的相似性
            direct2 = (x_wy).detach().cpu().numpy()
            dis = dis_fn(score_l_wy, score_l_woy, way=dis_way, direct1=direct1, direct2=direct2,
                         factor=param['cosine_factor'])
            dis_result, dis_index = dis.sort()
            dis_index = dis_index.tolist()

            # for cluster function
            # dis_index = cluster(x_woy.detach().cpu().numpy(),t_labels.detach().cpu().numpy().reshape(-1))

            # count the predict noise
            # for thr in [i*0.05 for i in range(0,int(FLAGS.noise//0.05)+1)]:
            test_labels = np.zeros(train_ds.shape[0])
            noise_count = sum(noise_index)
            n_s = [i for i in range(len(noise_index)) if noise_index[i]]  # noise sample's index

            if combine_class:
                threshold = make_threshold(dis_result, param['filter_factor'])
                noise_p_count = int(len(dis_result) * threshold)
                p_i = dis_index[-noise_p_count:]  # the noise index
            else:
                p_i = []
                dis_class_list = [[] for i in range(config_wy.data.output_size)]
                dis_class_re = [[] for i in range(config_wy.data.output_size)]

                for i in range(len(dis_result)):
                    item_label = torch.argmax(t_labels[dis_index[i]])
                    dis_class_list[item_label.item()].append(dis_index[i])
                    dis_class_re[item_label.item()].append(dis_result[i])

                for i in range(config_wy.data.output_size):
                    threshold = make_threshold(torch.tensor(dis_class_re[i]), param['filter_factor'])
                    noise_class_count = int(len(dis_class_list[i]) * threshold)
                    p_i.extend(dis_class_list[i][-noise_class_count:])

            if threshold == 0:
                p_i = []
            test_labels[n_s] = 1.
            # print(sum(test_labels),sum(predict))predict = np.zeros(train_ds.shape[0])
            # predict = np.zeros(train_ds.shape[0])
            # predict[p_i] = 1.
            # acc, precision, recall, f1 = cal_metric(test_labels, predict)
            # print(f'diff方法,precision{precision};recall:{recall};预测数量:{sum(predict)}')

            # from filter_easy import filter_loss
            # center = x_woy
            # p_i_add,center_model = filter_loss(center,t_labels)
            # predict = np.zeros(train_ds.shape[0])
            # predict[p_i_add] = 1.
            # acc, precision, recall, f1 = cal_metric(test_labels, predict)
            # print(f'loss方法,precision{precision};recall:{recall};预测数量:{sum(predict)}')

            from filter_easy import filter_nbrs
            p_i_add = filter_nbrs(x_woy, t_labels, n_neighbors=param['k_nbrs'], threshold=param['easy_fliter'])

            p_i.extend(p_i_add)
            p_i = list(set(p_i))
            predict = np.zeros(train_ds.shape[0])
            predict[p_i] = 1.
            predict = np.zeros(train_ds.shape[0])
            predict[p_i_add] = 1.
            # acc, precision, recall, f1 = cal_metric(test_labels, predict)
            # print(f'loss方法,precision{precision};recall:{recall};预测数量:{sum(predict)}')

            # acc, precision, recall, f1 = cal_metric(test_labels, predict)
            # print('all in all,')
            # print(" acc: %.5e" % (acc))
            # print(" precision: %.5e" % (precision))
            # print(" recall: %.5e" % (recall))
            # print(" f1: %.5e" % ( f1))
            # print('噪声阈值', threshold)
            # print("%.5e\t%.5e\t%.5e\t%.5e\t%d\t%d\t%d"%(acc,precision,recall,f1,len(set(p_i) - set(n_s)),noise_count,noise_p_count))
            # print(f'筛选噪声数:{sum(predict)}')
            # print('总噪声数', noise_count)
            # print('成功筛选', len(p_i) - len(set(p_i) - set(n_s)))
            # print('误判数:' + str(len(set(p_i) - set(n_s))))
            # print('漏判数:' + str(len(set(n_s) - set(p_i))))
            # print('成功筛选样本',list(set(p_i)-(set(p_i) - set(n_s))))
            # print('误判:' + str(sorted(list(set(p_i) - set(n_s)))))
            # print('漏判:' + str(sorted(list(set(n_s) - set(p_i)))))
            # print('噪声样本：' + str(sorted(n_s)))
            # print('噪声预测:' + str(sorted(p_i)))
            # suc_score = pd.DataFrame([{
            #     'noise_count': noise_count,
            #     'predict_count': len(p_i),
            #     'acc': acc,
            #     'precise': precision,
            #     'recall': recall
            # }])
            # remove predice noise item
            noise_data = torch.cat((train_ds, t_labels), dim=1).detach().cpu().numpy()

            # if del the noise data
            if len(p_i) == 0:
                cleaner_data = noise_data
            elif del_correct == 0:
                mask = np.ones(noise_data.shape[0], dtype=bool)
                mask[p_i] = False
                cleaner_data = noise_data[mask]
            elif del_correct == 1:
                # else correct the label
                #     cleaner_data[p_i, -config_wy.data.output_size:] = find_y(noise_data[p_i])
                # cleaner_data = noise_data
                # cleaner_data[p_i, -config_wy.data.output_size:] = find_y_by_classify(noise_data, p_i)
                pass
            elif del_correct == 2:
                if len(p_i) == train_ds.shape[0]:
                    avg_acc += 0
                    continue
                cleaner_data = integration_find(noise_data, p_i)

            else:
                # cleaner_data = instance_corr_del(noise_data,p_i,center,center_model)
                pass
            if cleaner_data.shape[0] != 0:
                cleaner_data = transformer.inverse_transform(cleaner_data)
                scores, _ = evaluation.compute_scores(train_ds_, eval_ds_, [cleaner_data], metadata=meta)

                # add_item = pd.concat([suc_score, scores.iloc[:, 1:]], axis=1)
                avg_acc += float(scores.iloc[0, 1])
                avg_precision += float(scores.iloc[0, 2])
                avg_recall += float(scores.iloc[0, 3])
                avg_f1 += float(scores.iloc[0, 4])
            else:
                avg_acc += 0
                avg_precision += 0
                avg_recall += 0
                avg_f1 += 0

        trial.set_user_attr("f1", avg_f1 / kf_times)
        trial.set_user_attr("recall", avg_recall / kf_times)
        trial.set_user_attr("precision", avg_precision / kf_times)
        return avg_acc / kf_times
        # downstream.append(add_item)
        # for col in add_item.columns:
        #     mutil_index.append((ckp_fdix, col))

        # multi_index = pd.MultiIndex.from_tuples(mutil_index, names=['fold', 'Column'])

        # downstream = pd.concat(downstream, axis=1)
        # downstream.columns = multi_index
        # # downstream = pd.concat([classify_name,downstream],  axis=1)  # add the classify name in the first column
        # downstream = pd.concat([pd.DataFrame([{'data_set': name}]), downstream],
        #                        axis=1)  # add the data name in the first row
        # file_path = './results.csv'

        # 如果文件不存在，写入数据并包含列索引名称
        # if not pd.io.common.file_exists(file_path):
        #     downstream.to_csv(file_path, index=False)
        # else:
        #     # 如果文件已存在，则追加数据，并且不写入列索引名称
        #     downstream.to_csv(file_path, mode='a', header=False, index=False)

    return objective


def main(avg):
    name = FLAGS.name
    config_flags.DEFINE_config_file("config_condy", 'configs/{}.py'.format(name), "Training configuration.",
                                    lock_config=True)
    flags.DEFINE_string("path_woy",
                        '{}/{}_woy_xscore/checkpoints/checkpoint_max.pth'.format(FLAGS.config_condy.training.noise,
                                                                                 name)
                        , "Training configuration.")
    flags.DEFINE_string("path_wy",
                        '{}/{}_condy_xscore/checkpoints/checkpoint_max.pth'.format(FLAGS.config_condy.training.noise,
                                                                                   name),
                        "Training configuration.")
    flags.DEFINE_string("path_classify",
                        '{}/{}_classify/checkpoints/checkpoint_max.pth'.format(FLAGS.config_condy.training.noise, name),
                        "Training configuration.")

    noise_list = [0.05, 0.1, 0.2, 0.4]
    noise_result = []
    for noise_rate in noise_list:
        FLAGS.config_condy.training.noise = noise_rate
        print('tackle the noise rage{}'.format(noise_rate))
        FLAGS.path_woy = '{}/{}_woy_xscore/checkpoints/checkpoint_max.pth'.format(FLAGS.config_condy.training.noise,
                                                                                  name)
        FLAGS.path_wy = '{}/{}_condy_xscore/checkpoints/checkpoint_max.pth'.format(FLAGS.config_condy.training.noise,
                                                                                   name)
        FLAGS.path_classify = '{}/{}_classify/checkpoints/checkpoint_max.pth'.format(FLAGS.config_condy.training.noise,
                                                                                     name)
        print(FLAGS.path_woy)
        global config_wy
        global config_woy
        global transformer
        global meta
        global sde
        global data
        global sampling_eps
        config_wy = copy.deepcopy(FLAGS.config_condy)
        config_woy = copy.deepcopy(config_wy)

        # for read data with y, if this is false will read data without y
        FLAGS.xscore_label = False  # if train the combine distribution p(x,y)
        FLAGS.cond_y = True  # if train the feature condition with y, set xscore_label==False
        mutils.after_defined(FLAGS, config_wy)

        FLAGS.xscore_label = False  # if train the combine distribution p(x,y)
        FLAGS.cond_y = False  # if train the feature condition with y, set xscore_label==False
        mutils.after_defined(FLAGS, config_woy)
        # model(x, t) t<1 ,t~[0.01,10]
        train_ds, eval_ds, (transformer, meta) = datasets.get_dataset(config_wy,
                                                                      uniform_dequantization=config_wy.data.uniform_dequantization)
        # Setup SDEs
        if config_wy.training.sde.lower() == 'vpsde':
            sde = sde_lib.VPSDE(beta_min=config_wy.model.beta_min, beta_max=config_wy.model.beta_max,
                                N=config_wy.model.num_scales)
            sampling_eps = 1e-3
        elif config_wy.training.sde.lower() == 'subvpsde':
            sde = sde_lib.subVPSDE(beta_min=config_wy.model.beta_min, beta_max=config_wy.model.beta_max,
                                   N=config_wy.model.num_scales)
            sampling_eps = 1e-3
        elif config_wy.training.sde.lower() == 'vesde':
            sde = sde_lib.VESDE(sigma_min=config_wy.model.sigma_min, sigma_max=config_wy.model.sigma_max,
                                N=config_wy.model.num_scales)
            sampling_eps = 1e-5
        else:
            raise NotImplementedError(f"SDE {config_wy.training.sde} unknown.")

        if meta['problem_type'] == 'binary_classification':
            metric = 'binary_f1'
        elif meta['problem_type'] == 'regression':
            metric = "r2"
        else:
            metric = 'macro_f1'
        data = np.concatenate((train_ds, eval_ds), axis=0)

        Model_list = [None]  # kf is range from 1 to 10
        for ix in range(1, kf_times + 1):
            path_woy = FLAGS.path_woy.replace('.pth', f"{ix}.pth")
            path_wy = FLAGS.path_wy.replace('.pth', f"{ix}.pth")
            path_classify = FLAGS.path_classify.replace('.pth', f"{ix}.pth")
            woy = get_socre(config_woy, path_woy)
            classify = get_class(config_wy, path_classify)
            wy = get_socre(config_wy, path_wy)
            Model_list.append({
                'woy': woy,
                'classify': classify,
                'wy': wy
            })
        objection = create_object(Model_list)

        study = optuna.create_study(direction='maximize')  # 最大化准确度

        # 优化超参数
        study.optimize(objection, n_trials=1)  # 执行100次试验
        print(study.best_trial.params)
        print(study.best_trial.value)
        best_para = study.best_trial.params
        best_para['precision'] = study.best_trial.user_attrs['precision']
        best_para['recall'] = study.best_trial.user_attrs['recall']
        best_para['f1'] = study.best_trial.user_attrs['f1']
        best_para['best_acc'] = study.best_trial.value
        best_para['name'] = name
        suc_score = pd.DataFrame([best_para])
        noise_result.append(suc_score)
    # 如果文件不存在，写入数据并包含列索引名称

        file_path = './parameter_set/{}-{}.csv'.format(noise_rate, name)
        if not pd.io.common.file_exists(file_path):
            suc_score.to_csv(file_path, index=False)
        else:
            # 如果文件已存在，则追加数据，并且不写入列索引名称
            suc_score.to_csv(file_path, mode='a', index=False)


if __name__ == "__main__":
    app.run(main)

