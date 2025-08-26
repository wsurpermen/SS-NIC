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


'''
for condition distribution p(x|y)
'''

FLAGS = flags.FLAGS
# 如果使用pycharm运行时，需要在路径前加../
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
flags.mark_flags_as_required(["name", "order",  "lam", "thr", "noise", "cond_y", "xscore_label"])

def show_score(model_wy,model_woy):
    ''' for two dimention '''
    import matplotlib.pyplot as plt
    # 计算每个特征的步长
    if isinstance(model_wy, torch.nn.DataParallel):  # for DataParaller
        model_wy = model_wy.module
        model_woy = model_woy.module
    num_points = 1000
    step = int(np.sqrt(num_points))  # 计算步数
    # 生成等间隔的数据点
    x = np.linspace(0, 1, step)
    y = np.linspace(0, 1, step)
    x, y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()
    for label in [None, 0, 1]:
        result_score = []
        for i in zip(x, y):
            if label is not None:
                result_score.append(model_wy(torch.tensor(i).to('cuda').float(),
                    torch.ones(1, device='cuda').float()*1e-5,
                                          torch.ones(1,device='cuda').float()*label).cpu().detach().numpy())
            else:
                result_score.append(model_woy(torch.tensor(i).to('cuda').float(),
                                          torch.ones(1, device='cuda').float() * 1e-5).cpu().detach().numpy())
        result_score = np.array(result_score)
        result_score = result_score.reshape(-1,2)
        u = result_score[:, 0]
        v = result_score[:, 1]

        speed = np.sqrt(u ** 2 + v ** 2)
        plt.quiver(x, y, u, v, speed, cmap='viridis')
        # 绘制数据点
        plt.colorbar()

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Equally Spaced Data Points')
        plt.grid(True)
        plt.savefig('score_result0{}.png'.format(label),dpi=1024)
        plt.clf()
    return 0

def make_cluster(x, dis):
    ''' x is a numpy'''
    cluster_list = []
    label_list = []
    for item in x:
        min_dis = np.inf
        min_index = 0
        # find the min dis with the cluster current or create a new cluster
        for label_index in range(len(cluster_list)):
            kind = cluster_list[label_index]
            center = kind.mean(axis=0)
            l2 = np.linalg.norm(item - center)
            if l2<dis and l2<min_dis:
                min_dis = l2
                min_index = label_index
        # if find a new cluster
        if min_dis < dis:
            cluster_list[min_index] = np.append(cluster_list[min_index], [item], axis=0)
            label_list.append(min_index)
        else:
            label_list.append(len(cluster_list))
            cluster_list.append(np.array([item]))
    return np.array(label_list)


def read_csv_params(noise_ratio, data_name, dir="./parameter_set/"):
    # 从文件名提取噪音比和数据名称
    filepath = dir + str(noise_ratio) + '-' + data_name + '.csv'
    # 读取CSV文件
    df = pd.read_csv(filepath)

    # 将每列存入变量
    cosine_factor = df['cosine_factor'].iloc[0]
    filter_factor = df['filter_factor'].iloc[0]
    time_step = df['time_step'].iloc[0]
    classify_lambda = df['classify_lambda'].iloc[0]
    k_nbrs = df['k_nbrs'].iloc[0]
    easy_fliter = df['easy_fliter'].iloc[0]
    precision = df['precision'].iloc[0]
    recall = df['recall'].iloc[0]
    f1 = df['f1'].iloc[0]
    best_acc = df['best_acc'].iloc[0]
    name = df['name'].iloc[0]

    return {
        'cosine_factor': cosine_factor,
        'filter_factor': filter_factor,
        'time_step': time_step,
        'classify_lambda': classify_lambda,
        'k_nbrs': k_nbrs,
        'easy_fliter': easy_fliter,

    }

def cosine_distance(vec1, vec2):
    """计算两个向量之间的余弦距离"""
    dot_product = np.sum(vec1 * vec2, axis=1)
    norm1 = np.linalg.norm(vec1, axis=1)
    norm2 = np.linalg.norm(vec2, axis=1)
    return 1 - dot_product / (norm1 * norm2)

def cluster(x, y):
    # 使用 OPTICS 进行聚类
    optics = OPTICS(min_samples=3, max_eps=0.5, metric='euclidean')
    labels = optics.fit_predict(x)

    # 获取所有簇标签
    unique_labels = np.unique(labels)

    # 初始化一个字典来存储每个簇的质心
    centroids = {}
    print('labels==-1')
    print(sum(labels == -1))
    # 计算每个簇的质心
    r_bool = np.array([False for i in range(len(y))])
    for label in unique_labels:
        if label == -1:
            continue
        # 获取当前簇的所有样本
        y_point = y[labels == label]
        # 计算每个元素的出现次数
        unique_elements, counts = np.unique(y_point, return_counts=True)
        print(unique_elements)
        if len(unique_elements) == 1: continue
        # 找到出现次数的最大
        max_count = np.max(counts)
        # 找到出现次数不等于最大值的元素
        frequent_elements = unique_elements[counts != max_count]
        for i in frequent_elements:
            r_bool = np.logical_or(r_bool, np.logical_and(y == i, labels == label))
    noise_index = np.where(r_bool)[0]

    return noise_index.tolist()

def make_threshold(tensor,factor=0.8):
    way = 0
    t = 0
    if way == 0 : # base on IQR
        a = torch.median(tensor)  + (torch.quantile(tensor,0.75)- torch.quantile(tensor,0.25))*factor
        for i in range(tensor.shape[0]-1, 0,-1):
            if tensor[i] < a:
                t = i
                break

    elif way ==1: # base on z-score
        mean = torch.mean(tensor)
        std = torch.std(tensor)

        # 计算 z-score
        z_scores = (tensor - mean) / std
        for i in range(z_scores.shape[0] - 1, 0, -1):
            if z_scores[i] < 1.5:
                t = i
                break
    return ((tensor.shape[0] - t) / tensor.shape[0])


def main(avg):
    def find_y(item):
        item = item.astype('float32')
        item_list = []
        semi_x = torch.tensor(item[:, :-config_wy.data.output_size]).to(config_wy.device)
        for y_bais in range(0, config_wy.data.output_size):
            _y_b = (np.argmax(item[:, -config_wy.data.output_size:], axis=1) + y_bais)
            _y_b %= config_wy.data.output_size
            one_hot_y = np.zeros((len(_y_b), config_wy.data.output_size))
            one_hot_y[np.arange(len(_y_b)), _y_b] = 1
            one_hot_y = one_hot_y.astype('float32')

            _, score_temp1 = sampling_fn_wy(model_wy, sampling_shape=(semi_x.shape[0], config_wy.data.image_size),
                                           distance=True,
                                           semi_x=semi_x,
                                           labels=torch.tensor(one_hot_y).to(config_wy.device),
                                              timestep=FLAGS.timestep)

            _, score_temp2 = sampling_fn_woy(model_wy, sampling_shape=(semi_x.shape[0], config_wy.data.image_size),
                                           distance=True,
                                           semi_x=semi_x,
                                           labels=torch.tensor(one_hot_y).to(config_wy.device),
                                                 timestep=FLAGS.timestep, xscore_sample=True,
                                                 xmodle=model_woy)
            dis_= dis_fn(score_temp1,score_temp2).view(-1, 1).detach().cpu().numpy()
            item_list.append(dis_)
        item_list = np.concatenate(item_list, axis=1)
        one_hot_y = np.zeros((len(item), config_wy.data.output_size))
        one_hot_y[np.arange(len(item)), np.argmax(item_list, axis=1)] = 1

        diff =np.sum(np.argmax(item[:,-config_wy.data.output_size:], axis=1) != np.argmax(item_list,axis=1))
        logging.info(f'纠正标签个数:{diff}')
        return one_hot_y

    def find_y_by_classify(noise_data,p_i):
        # p_i is the noise index predict
        predict = np.ones(noise_data.shape[0],dtype='bool')
        predict[p_i] = False
        cleaner_data = noise_data[predict]
        np_feature = cleaner_data[:,:-config_wy.data.output_size]
        np_label = np.argmax(cleaner_data[:,-config_wy.data.output_size:],axis=1)
        un_data = noise_data[~predict]
        un_feature = un_data[:,:-config_wy.data.output_size]
        # un_label = un_data[:,-config_wy.data.output_size:]
        if len(np.unique(np_label)) == 1 :
            return un_data[:,-config_wy.data.output_size:]

        # for c4.5
        clf = DecisionTreeClassifier(criterion='entropy', random_state=42)  # 'entropy' 近似 C4.5
        clf.fit(np_feature, np_label)
        y_pred1 = clf.predict(un_feature)


        # for LOG
        log_reg = LogisticRegression(penalty='l2', C=1e8, solver='lbfgs', multi_class='auto', random_state=42)
        log_reg.fit(np_feature, np_label)
        y_pred2 = log_reg.predict(un_feature)

        # for 3-nn ,这个太卡了
        knn = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
        knn.fit(np_feature, np_label)
        y_pred3 = knn.predict(un_feature)
        # 创建一个新数组，设置一致的地方为 0，不一致的地方为 1

        pred = np.column_stack((y_pred1,y_pred2,y_pred3))
        final_predictions = mode(pred, axis=1).mode.flatten()
        one_hot_y = np.zeros((len(un_feature), config_wy.data.output_size))
        one_hot_y[np.arange(len(un_feature)), final_predictions] = 1
        return one_hot_y

    def instance_corr_del(noise_data,p_i,center,model,threshold=0.8):
        # p_i is the noise index predict
        predict = np.ones(noise_data.shape[0], dtype='bool')
        predict[p_i] = False
        cleaner_data = noise_data[predict]

        np_label = np.argmax(cleaner_data[:, -config_wy.data.output_size:], axis=1)

        un_data = noise_data[~predict]
        un_feature = un_data[:, :-config_wy.data.output_size]
        # un_label = un_data[:,-config_wy.data.output_size:]
        if len(np.unique(np_label)) == 1:
            return noise_data

        model.eval()
        center_pre_logits = model(center[~predict]).detach().cpu().numpy()

        mlp = MLPClassifier(hidden_layer_sizes=(int(config_wy.data.output_size*3/2),), activation='relu', max_iter=200, random_state=42)
        # 训练模型
        mlp.fit(cleaner_data[:,:-config_wy.data.output_size], cleaner_data[:,-config_wy.data.output_size:])
        mlp_prd_logits = mlp.predict_proba(un_feature)

        def softmax(logits):
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # 防止溢出
            return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # 计算 softmax 后的概率
        softmax_1 = mlp_prd_logits
        softmax_2 = softmax(center_pre_logits)


        # 找到每个样本的最大 logits 所对应的类别
        pred_1 = np.argmax(softmax_1, axis=1)  # 获取最大值的索引（类别标签）
        pred_2 = np.argmax(softmax_2, axis=1)

        same_class_and_confident = (pred_1 == pred_2) & (np.max(softmax_1, axis=1) > threshold)& (np.max(softmax_2, axis=1) > threshold)
        one_hot_y = np.zeros((len(un_feature), config_wy.data.output_size))
        one_hot_y[range(len(un_feature)),pred_1] = 1.0
        filter_feature = un_feature[same_class_and_confident]
        filter_label = one_hot_y[same_class_and_confident]
        sub_set = np.concatenate((filter_feature,filter_label),axis=1)

        return np.concatenate((cleaner_data,sub_set),axis=0)

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

        if len(unique_labels) == 1:
            one_hot_y = np.zeros(config_wy.data.output_size)
            one_hot_y[y_train[0]] = 1
            return np.array([one_hot_y for i in range(len(p_i))])
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
        # return cleaner_data # w/o correct    not annotate to ignore the correct method
        return np.concatenate(((cleaner_data,un_data[consistent_predictions])), axis=0)

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

    combine_class = False # false to filter class by class
    del_correct = 2 # false to correct else del
    dis_way = 10
    factor = 0.8

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



    config_wy = copy.deepcopy(FLAGS.config_condy)

    config_woy = copy.deepcopy(config_wy)

    params = read_csv_params(config_wy.training.noise, name)

    cosine_factor = params['cosine_factor']
    # cosine_factor = 0 # w/o cosine
    filter_factor = params['filter_factor']
    time_step = params['time_step']
    classify_lambda = params['classify_lambda']
    k_nbrs = params['k_nbrs']
    easy_fliter = params['easy_fliter']



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
    ckp_fdix = 0
    downstream = [None]  # the None is a place for name of classify


    mutil_index = []
    for train_index, val_index in kf.split(data):

        # set randomSeed
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
        path_woy = FLAGS.path_woy.replace('.pth',f"{ckp_fdix}.pth")
        path_wy = FLAGS.path_wy.replace('.pth',f"{ckp_fdix}.pth")
        path_classify = FLAGS.path_classify.replace('.pth', f"{ckp_fdix}.pth")

        eval_ds_ = transformer.inverse_transform(eval_ds)   # for downstream task
        train_ds_ =  transformer.inverse_transform(train_ds)

        np.random.seed(randomSeed)
        real_labels = train_ds[:, -config_wy.data.output_size:]
        train_ds = train_ds[:, :-config_wy.data.output_size]

        t_labels = real_labels.copy()

        # cluster_index = train_ds[:,-1]
        # filtered_rows = train_ds[cluster_index < 1]
        # cluster(filtered_rows)

        # add noise for label
        t_labels, noise_index = add_label_noise(t_labels,config_wy)

        '''
        # for synthesize data
        l = 200
        rnnn = 10
        for i in range(l-rnnn,l):noise_index[i]= 1
        for i in range(2*l-rnnn,2*l):noise_index[i]= 1
        '''

        train_ds = torch.tensor(train_ds).to(config_wy.device).float()
        t_labels = torch.tensor(t_labels).to(config_wy.device).float()

        # Setup SDEs
        if config_wy.training.sde.lower() == 'vpsde':
            sde = sde_lib.VPSDE(beta_min=config_wy.model.beta_min, beta_max=config_wy.model.beta_max, N=config_wy.model.num_scales)
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

        # model_woy can be used as p(x|y) if input the sampling_fn label
        model_woy = get_socre(config_woy, path_woy)
        inverse_scaler_woy = datasets.get_data_inverse_scaler(config_woy)
        sampling_shape_woy = (eval_ds.shape[0], config_woy.data.image_size)
        sampling_fn_woy = sampling.get_sampling_fn(config_woy, sde, sampling_shape_woy, inverse_scaler_woy, sampling_eps)

        # init model with lable y
        # config.data.image_size += 1   # because the label y without one-hot encoder
        #model_wy = get_socre(config_wy, path_wy)
        inverse_scaler_wy = datasets.get_data_inverse_scaler(config_wy)
        sampling_shape_wy = (eval_ds.shape[0], config_wy.data.image_size)
        sampling_fn_wy = sampling.get_sampling_fn(config_wy, sde, sampling_shape_wy, inverse_scaler_wy, sampling_eps)

        # model: logp(x|y) + logp(y|x)'
        model_wy = get_s_c(config_wy, path_wy, path_classify, eval= True, sde=sde, condition=True,lambda_=classify_lambda)
        # model_wy = get_s_c(config_woy, path_woy, path_classify, eval= True, sde=sde, condition=False)
        # show_score(model_wy,model_woy)
        # return 0

        # if input mudel is p(x,y), need cut down the model's output

        x_woy, score_l_woy = sampling_fn_woy(model_woy, sampling_shape=sampling_shape_wy, distance=True,
                                             semi_x=train_ds,
                                             timestep=time_step, condition_sample=False)

        x_wy, score_l_wy = sampling_fn_wy(model_wy, sampling_shape=sampling_shape_wy, distance=True,
                                          semi_x=train_ds,
                                          labels=t_labels,
                                          timestep=time_step, condition_sample=True)
        direct1 = (x_woy).detach().cpu().numpy()  # 计算最终两个收敛点的相似性
        direct2 = (x_wy).detach().cpu().numpy()

        dis = dis_fn(score_l_wy, score_l_woy,way=dis_way,direct1=direct1,direct2=direct2,factor=cosine_factor)
        dis_result, dis_index = dis.sort()
        dis_index = dis_index.tolist()

        # for cluster function
        # dis_index = cluster(x_woy.detach().cpu().numpy(),t_labels.detach().cpu().numpy().reshape(-1))

        # count the predict noise
        # for thr in [i*0.05 for i in range(0,int(FLAGS.noise//0.05)+1)]:


        n_s = [i for i in range(len(noise_index)) if noise_index[i]]   # noise sample's index


        if combine_class:
            threshold = make_threshold(dis_result, filter_factor)
            noise_p_count = int(len(dis_result)*threshold)
            p_i = dis_index[-noise_p_count:]  # the noise index
        else:
            p_i = []
            dis_class_list = [[] for i in range(config_wy.data.output_size)]
            dis_class_re = [[] for i in range(config_wy.data.output_size)]

            for i in range(len(dis_result)):
                item_label =  torch.argmax(t_labels[dis_index[i]])
                dis_class_list[item_label.item()].append(dis_index[i])
                dis_class_re[item_label.item()].append(dis_result[i])

            for i in range(config_wy.data.output_size):
                threshold = make_threshold(torch.tensor(dis_class_re[i]),filter_factor)
                noise_class_count = int(len(dis_class_list[i])*threshold)
                p_i.extend(dis_class_list[i][-noise_class_count:])

        test_labels = np.zeros(train_ds.shape[0])
        noise_count = sum(noise_index)
        if threshold ==0:
            p_i = []
        test_labels[n_s] = 1.
        #print(sum(test_labels),sum(predict))predict = np.zeros(train_ds.shape[0])
        predict = np.zeros(train_ds.shape[0])
        predict[p_i] = 1.
        acc, precision, recall, f1 = cal_metric(test_labels, predict)
        print(f'diff方法,precision{precision};recall:{recall};预测数量:{sum(predict)}')

        # from filter_easy import filter_loss
        # center = x_woy
        # p_i_add,center_model = filter_loss(center,t_labels)
        # predict = np.zeros(train_ds.shape[0])
        # predict[p_i_add] = 1.
        # acc, precision, recall, f1 = cal_metric(test_labels, predict)
        # print(f'loss方法,precision{precision};recall:{recall};预测数量:{sum(predict)}')

        from filter_easy import filter_nbrs
        p_i_add = filter_nbrs(x_woy,t_labels,n_neighbors=k_nbrs,threshold=easy_fliter)
        predict = np.zeros(train_ds.shape[0])
        predict[p_i_add] = 1.
        acc, precision, recall, f1 = cal_metric(test_labels, predict)
        print(f'neigthber方法,precision{precision};recall:{recall};预测数量:{sum(predict)}')


        p_i.extend(p_i_add)       # w/o neighbor ,  annotation to ignore the neighbor method
        # p_i = p_i_add            # w/o score_dis, not annotation to ignore the score method
        p_i = list(set(p_i))
        predict = np.zeros(train_ds.shape[0])
        predict[p_i] = 1.
        acc, precision, recall, f1 = cal_metric(test_labels, predict)

        #print('all in all,')
        print(" acc: %.5e" % ( acc))
        print(" precision: %.5e" % ( precision))
        print(" recall: %.5e" % ( recall))
        #print(" f1: %.5e" % ( f1))
        # print('噪声阈值', threshold)
        #print("%.5e\t%.5e\t%.5e\t%.5e\t%d\t%d\t%d"%(acc,precision,recall,f1,len(set(p_i) - set(n_s)),noise_count,noise_p_count))
        print(f'筛选噪声数:{sum(predict)}' )
        print('总噪声数', noise_count)
        print('成功筛选', len(p_i)-len(set(p_i) - set(n_s)))
        # print('误判数:' + str(len(set(p_i) - set(n_s))))
        #print('漏判数:' + str(len(set(n_s) - set(p_i))))
        # print('成功筛选样本',list(set(p_i)-(set(p_i) - set(n_s))))
        # print('误判:' + str(sorted(list(set(p_i) - set(n_s)))))
        # print('漏判:' + str(sorted(list(set(n_s) - set(p_i)))))
        # print('噪声样本：' + str(sorted(n_s)))
        # print('噪声预测:' + str(sorted(p_i)))

        # remove predice noise item
        noise_data = torch.cat((train_ds,t_labels), dim=1).detach().cpu().numpy()

        # if del the noise data
        if len(p_i) ==0:
            cleaner_data = noise_data
        elif del_correct ==0:
            mask = np.ones(noise_data.shape[0], dtype=bool)
            mask[p_i] = False
            cleaner_data = noise_data[mask]
        elif del_correct ==1:
        # else correct the label
        #     cleaner_data[p_i, -config_wy.data.output_size:] = find_y(noise_data[p_i])
            cleaner_data = noise_data
            cleaner_data[p_i, -config_wy.data.output_size:] = find_y_by_classify(noise_data,p_i)
        elif del_correct == 2:
            cleaner_data = integration_find(noise_data, p_i)
        else:
            # cleaner_data = instance_corr_del(noise_data,p_i,center,center_model)
            pass
        cleaner_data = transformer.inverse_transform(cleaner_data)
        scores, _ = evaluation.compute_scores(train_ds_, eval_ds_, [cleaner_data], metadata = meta)

        suc_score = pd.DataFrame([{
            f'{ckp_fdix}noise_count': sum(noise_index),
            f'{ckp_fdix}predict_count': sum(predict),
            f'{ckp_fdix}acc': acc,
            f'{ckp_fdix}precise': precision,
            f'{ckp_fdix}recall': recall,
            f'{ckp_fdix}down_acc': scores.iloc[:, 1:].values[0][0],
            f'{ckp_fdix}down_precision': scores.iloc[:, 1:].values[0][1],
            f'{ckp_fdix}down_recall': scores.iloc[:, 1:].values[0][2],
            f'{ckp_fdix}down_f1': scores.iloc[:, 1:].values[0][3]
        }])
        downstream.append(suc_score)
    downstream = pd.concat(downstream, axis=1)

    # downstream = pd.concat([classify_name, downstream], axis=1)  # add the classify name in the first column
    downstream = pd.concat([pd.DataFrame([{'data_set': name}]), downstream],
                           axis=1)  # add the data name in the first row
    file_path = './results.csv'

# 如果文件不存在，写入数据并包含列索引名称
    if not pd.io.common.file_exists(file_path):
        downstream.to_csv(file_path, index=False)
    else:
    # 如果文件已存在，则追加数据，并且不写入列索引名称
        downstream.to_csv(file_path, mode='a', header=False, index=False)

if __name__ == "__main__":
    app.run(main)

