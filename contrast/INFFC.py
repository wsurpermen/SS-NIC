from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import gc


def post_prune(tree, confidence_factor=0.25):
    # 后剪枝
    # 获取树的节点信息
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    value = tree.tree_.value

    # 递归剪枝
    def prune_index(index):
        if children_left[index] == children_right[index]:  # 如果是叶子节点，返回
            return

        if (tree.tree_.impurity[children_left[index]] == 0 and
            tree.tree_.impurity[children_right[index]] == 0):
            # 如果左右子节点都是纯节点，考虑剪枝
            left_count = np.sum(value[children_left[index]])
            right_count = np.sum(value[children_right[index]])
            total_count = left_count + right_count

            # 计算错误率
            error_rate = min(left_count, right_count) / total_count

            # 计算置信区间
            z = 1.96  # 95% 置信水平
            delta_error = z * np.sqrt(error_rate * (1 - error_rate) / total_count)

            # 如果误差减少不显著，则剪枝
            if error_rate + delta_error <= confidence_factor:
                # 将当前节点变为叶子节点
                feature[index] = -2
                threshold[index] = -2
                value[index] = value[children_left[index]] + value[children_right[index]]

        else:
            prune_index(children_left[index])
            prune_index(children_right[index])

    prune_index(0)  # 从根节点开始剪枝

# 应用后剪枝


def FC(np_feature, np_label,filter_np):
    # the input feature and label should be numpy
    # if filter_np is False ,represent the first step
    # return a nool of numpy, true represent the noise and false represent clean

    # for c4.5
    clf = DecisionTreeClassifier(criterion='entropy',random_state=42)  # 'entropy' 近似 C4.5
    clf.fit(np_feature[filter_np], np_label[filter_np])
    post_prune(clf, confidence_factor=0.25)
    y_pred = clf.predict(np_feature)
    comparison_result2 = np.where(y_pred == np_label, 0, 1)


    # for LOG
    log_reg = LogisticRegression(penalty='l2', C=1e8, solver='lbfgs', multi_class='auto', random_state=42)
    log_reg.fit(np_feature[filter_np], np_label[filter_np])
    y_pred = log_reg.predict(np_feature)
    comparison_result3 = np.where(y_pred == np_label, 0, 1)


    # for 3-nn ,这个太卡了
    knn = KNeighborsClassifier(n_neighbors=3,algorithm='ball_tree')
    knn.fit(np_feature[filter_np], np_label[filter_np])
    y_pred = knn.predict(np_feature)
    # 创建一个新数组，设置一致的地方为 0，不一致的地方为 1
    comparison_result1 = np.where(y_pred == np_label, 0, 1)
    

    return np.where((comparison_result1+comparison_result2+comparison_result3)>1,True,False)




def noisescore(np_feature, np_label, predict,k=6):
    # in predict, true represent the noise and false represent clean
    #  despite k == 6, but the nieghbors will return a set that include itself, so need remove it self
    def _t_e(index):
        times = 0
        for i in range(len(predict)):
            if predict[i] and i!=index:
                indices = nbrs.kneighbors(np_feature[i].reshape(1, -1),return_distance=False)[0][1:]

                if i in indices:
                    times += 1
        return times

    def _n_e(index):
        number = 0
        indices = nbrs.kneighbors(np_feature[index].reshape(1, -1),return_distance=False)[0][1:]
        for i in indices:
            if predict[i]:
                number +=1
        return number

    def _clean(index):
        add_item = _n_e(index) - k
        if predict[index]:
            return (k+add_item)/2/k
        else:
            return (k-add_item)/2/k


    def _confidence(index):
        return 1/((1+_t_e(index)**2)**0.5)

    def _neighborhood(index):
        indices = nbrs.kneighbors(np_feature[index].reshape(1, -1),return_distance=False)[0][1:]
        nbd = 0
        for neigh_i in indices:
            add_item = _clean(neigh_i)*_confidence(neigh_i)
            if np_label[index] == np_label[neigh_i]:
                nbd -= add_item
            else:
                nbd += add_item
        return nbd/k


    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(np_feature)
    # predict is a bool np, true represent the noise
    ns = np.zeros_like(predict,dtype=float)-1  # save the noise score
    for i in range(len(predict)):
        if predict[i]:
            ns[i] = _confidence(i)*_neighborhood(i)
            
    del nbrs
    gc.collect()
    return ns


def run(train_ds,t_labels_num,eval_ds,eval_label,config_wy,train_ds_,eval_ds_,meta,threshod=0,g=3,p=0.01):
    gc.collect()
    np_feature = train_ds.detach().cpu().numpy()
    np_label = t_labels_num.detach().long().cpu().numpy()
    g_add = 0
    filter_np = np.zeros_like(np_label).astype(bool)
    while 1:
        gc.collect()
        # first step
        preliminary = FC(np_feature, np_label, ~filter_np)
        # second step
        noise_indices = FC(np_feature, np_label, ~preliminary)
        # noise remove
        ns = noisescore(np_feature,np_label,noise_indices)
        ns_filter = ns > threshod

        difference_count = np.sum(ns_filter != filter_np)
        filter_np = ns_filter | filter_np
        if difference_count/len(t_labels_num)<=p:
            g_add +=1
            if g_add >=g:
                return filter_np
        else:
            g_add =0





