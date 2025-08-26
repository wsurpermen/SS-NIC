import math

from numpy import *
import numpy as np
from collections import Counter
from random import choice
import warnings
import time
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

minNumSample = 2


class BinaryTree:
    def __init__(self, labels=array([]), datas=array([])):
        self.label = labels
        self.data = datas
        self.leftChild = None
        self.rightChild = None

    def set_rightChild(self, rightObj):
        self.rightChild = rightObj

    def set_leftChild(self, leftObj):
        self.leftChild = leftObj

    def get_rightChild(self):
        return self.rightChild

    def get_leftChild(self):
        return self.leftChild

    def get_data(self):
        return self.data

    def get_label(self):
        return self.label


def deleteNoiseData(data, noiseOrder):
    # return a bool, ture reprensent noiseOrder ==0
    m, n = data.shape
    data = hstack((data, noiseOrder.reshape(m, 1)))
    redata = array(list(filter(lambda x: x[n] == 0, data[:, ])))
    if len(redata)==0:
        return redata
    redata = delete(redata, n, axis=1)
    return redata


def checkLabelSequence(labels):
    index1 = 0
    leaf_label = labels[0]
    for i in range(1, len(labels)):
        if leaf_label != labels[i]:
            index1 = i
            break
    if index1 == 0:
        return 0

    index2 = 0
    for i in range(index1 + 1, len(labels)):
        if leaf_label == labels[i]:
            index2 = i - 1
            break
    if index2 == 0:
        index2 = len(labels)
    return index2 - index1


def visitCRT(tree):  # Afferent root node
    if not tree.get_leftChild() and not tree.get_rightChild():
        data = tree.get_data()
        labels = checkLabelSequence(tree.get_label())

        try:
            labels = zeros(len(data)) + labels
        except TypeError:
            pass
        result = vstack((data, labels))
        return result
    else:
        resultLeft = visitCRT(tree.get_leftChild())
        resultRight = visitCRT(tree.get_rightChild())
        result = hstack((resultLeft, resultRight))
        return result



def splitData(data, splitAttribute, splitValue):
    rightData = array(list(filter(lambda x: x[splitAttribute] > splitValue, data[:, ])))
    leftData = array(list(filter(lambda x: x[splitAttribute] <= splitValue, data[:, ])))
    return leftData, rightData


def generateTree(data, uplabels=[]):
    try:
        numberSample, numberAttribute = data.shape
    except ValueError:
        numberSample = 1
        numberAttribute = data.size

    if numberAttribute == 0:
        return None

    numberAttribute = numberAttribute - 2

    labelNumKey = []
    if numberSample == 1:
        labelvalue = data[0][0]
        rootdata = data[0][numberAttribute + 1]
    else:
        labelNum = Counter(data[:, 0])
        labelNumKey = list(labelNum.keys())
        labelNumValue = list(labelNum.values())
        labelvalue = labelNumKey[labelNumValue.index(max(labelNumValue))]
        rootdata = data[:, numberAttribute + 1]
    rootlabel = hstack((labelvalue, uplabels))

    CRTree = BinaryTree(rootlabel, rootdata)

    if numberSample < minNumSample or len(labelNumKey) < 2:  # minNumSample The default is 10 or there is only 1 type of tag left
        return CRTree
    else:
        maxCycles = 1.5 * numberAttribute  # Maximum number of cycles
        i = 0
        while True:
            i += 1
            splitAttribute = random.randint(1, numberAttribute)
            if splitAttribute > 0 and splitAttribute < numberAttribute + 1:
                dataSplit = data[:, splitAttribute]
                uniquedata = list(set(dataSplit))
                if len(uniquedata) > 1:
                    break
            if i > maxCycles:
                return CRTree
        sv1 = random.choice(uniquedata)
        i = 0
        while True:
            i += 1
            sv2 = random.choice(uniquedata)
            if sv2 != sv1:
                break
            if i > maxCycles:
                return CRTree
        splitValue = mean([sv1, sv2])

        leftdata, rightdata = splitData(data, splitAttribute, splitValue)

        CRTree.set_leftChild(generateTree(leftdata, rootlabel))
        CRTree.set_rightChild(generateTree(rightdata, rootlabel))
        return CRTree


# Add the ordinal column at the end and call the function generateTree spanning tree
def CRT(data):
    numberSample, numberAttribute = data.shape
    orderAttribute = arange(numberSample).reshape(numberSample, 1)  # (646, 1)
    data = hstack((data, orderAttribute))  # Add another column at the end
    completeRandomTree = generateTree(data)
    return completeRandomTree



def Compare(save_best, best_parament):
    if save_best == []:
        save_best = best_parament.copy()
    else:
        if save_best[2] < best_parament[2]:  # Choose the one with the highest accuracy first
            save_best = best_parament.copy()
        elif save_best[2] == best_parament[2]:  # Compare the number of trees with the same precision to select trees with fewer trees
            if save_best[1] > best_parament[1]:
                save_best = best_parament.copy()
            elif save_best[1] == best_parament[1]:  # Compare the number of grains at the same time to select the threshold with a small threshold
                if save_best[0] > best_parament[0]:
                    save_best = best_parament.copy()
    return save_best


def Compare_and_noise(noise_data, noise_data_return, save_best, best_parament):
    if save_best == []:
        save_best = best_parament.copy()
        noise_data_return = noise_data
    else:
        if save_best[2] < best_parament[2]:  # Choose the one with the highest accuracy first
            save_best = best_parament.copy()
            noise_data_return = noise_data
        elif save_best[2] == best_parament[2]:  # Compare the number of trees with the same precision to select trees with fewer trees
            if save_best[1] > best_parament[1]:
                save_best = best_parament.copy()
                noise_data_return = noise_data
            elif save_best[1] == best_parament[1]:  # Compare the number of grains at the same time to select the threshold with a small threshold
                if save_best[0] > best_parament[0]:
                    save_best = best_parament.copy()
                    noise_data_return = noise_data
    return noise_data_return, save_best


def CRFNFL_adp(classifier, traindata, val, ntree=40, niThreshold=11):
    best_parament = [0, 0, 0, 0]
    save_best = []
    save_noise = []
    if ntree < 1:
        print('The value of ntree at least is 1.')
        return 0
    try:
        m, n = traindata.shape
    except ValueError as e:
        print(str(e))
        return 0

    forest = array([])
    for i in range(ntree):
        tree = CRT(traindata)
        visiTree = visitCRT(tree)
        visiTree = visiTree[:, argsort(visiTree[0, :])]
        visiTree = visiTree[1, :]
        if forest.size == 0:
            forest = visiTree.reshape(m, 1)  # m column, 1 line
        else:
            forest = hstack((forest, visiTree.reshape(m, 1)))  # hstack：Increase the column  vstack：Increase the line
    noise_data_return = zeros(m)
    time_start_crf = time.perf_counter()

    for subNi in range(2, niThreshold + 1):
        noiseForest = zeros(m)
        for j in range(m):
            for k in range(ntree):
                if forest[j, k] >= subNi:
                    noiseForest[j] += 1
                # print( noiseForest[j])
            if noiseForest[j] >= 0.5 * ntree:  # votes
                noiseForest[j] = 1
            else:
                noiseForest[j] = 0

        noise_data = noiseForest
        save_noise.append(sum(noise_data))
        denoiseTraindata = deleteNoiseData(traindata, noiseForest)
        try:
            best_parament[2] = Func(classifier, denoiseTraindata, val)  # Basic classifier classification results
        except:
            best_parament[2] = 0
        best_parament[0] = subNi
        best_parament[1] = ntree
        best_parament[3] = sum(noise_data)
        noise_data_return, save_best = Compare_and_noise(noise_data, noise_data_return, save_best,
                                                         best_parament)

    time_CRF = time.perf_counter() - time_start_crf
    time_start_kmeans = time.perf_counter()
    X = np.array(save_noise)
    X = X.reshape(-1, 1)

    kmeans_label = KMeans(n_clusters=2).fit(X).labels_
    tmp_noise = 0
    for i in range(1, len(kmeans_label)):
        if kmeans_label[i] != kmeans_label[i - 1]:
            tmp_noise = i + 2
    time_Kmeans = time.perf_counter() - time_start_kmeans

    noise_new = zeros(m)
    for j in range(m):
        for k in range(ntree):
            if forest[j, k] >= tmp_noise:
                noise_new[j] += 1

        if noise_new[j] >= 0.5 * ntree:  # votes
            noise_new[j] = 1
        else:
            noise_new[j] = 0
    save_best.append(time_CRF)
    return noise_data_return, save_best, save_noise, time_Kmeans, tmp_noise, noise_new


def Func(classifier, traindata, testdata):
    traindatalabel = traindata[:, 0]
    traindata = traindata[:, 1:]
    testdatalabel = testdata[:, 0]
    testdata = testdata[:, 1:]
    classifier.fit(traindata, traindatalabel)
    return classifier.score(testdata, testdatalabel)  # Prediction accuracy


def crfnfl_adp(classifier, traindata):
    save_best = []
    dic = {}
    dic_kmeans = {}
    save_noise = []
    time_Kmeans = 0
    NI_kmeans = []
    for i in range(len(traindata)):
        dic[i] = 0
        dic_kmeans[i] = 0
    ni_Threshold = math.ceil(math.log(len(traindata), 2))
    folds = KFold(n_splits=10, shuffle=True)
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(traindata)):
        trn = traindata[trn_idx]
        val = traindata[val_idx]
        noise, best_parament, save_noise, time_Kmeans, tmp_noise, noise_kmeans = CRFNFL_adp(classifier, trn, val,
                                                                                            niThreshold=ni_Threshold)
        NI_kmeans.append(tmp_noise)
        save_best = Compare(save_best, best_parament)
        for i in range(len(trn_idx)):
            dic[trn_idx[i]] += noise[i]
            dic_kmeans[trn_idx[i]] += noise_kmeans[i]
    a = []
    b = []
    tmp_noise = max(NI_kmeans, key=NI_kmeans.count)
    for i in dic.keys():
        if dic[i] < 5:
            a.append(i)
    for i in dic_kmeans.keys():
        if dic_kmeans[i] < 5:
            b.append(i)
    a = array(a)

    b = array(b)
    train = traindata[a] if len(a)!=0 else array([])
    predict = np.ones(traindata.shape[0], dtype=bool)
    if b.shape[0] !=0:
        train_kmeans = traindata[b]
        predict[b] = False
    else:
        train_kmeans = b
    # train is the VCV, train_kmeans is the AM, predict is the bool np of kmeans whic one represent noise
    return train, save_best, save_noise, time_Kmeans, tmp_noise, train_kmeans, predict

def add_flip_noise(dataset, noise_rate):
    label_cat = list(set(dataset[:, 0]))
    flag = 0
    for i in range(len(label_cat)):
        label = label_cat[i]
        other_label = list(filter(lambda x: x != label, label_cat))
        data = dataset[dataset[:, 0] == label]
        n = data.shape[0]
        noise_num = int(n * noise_rate)
        noise_index_list = []
        n_index = 0
        while True:
            rand_index = int(random.uniform(0, n))
            if rand_index in noise_index_list:
                continue
            if n_index < noise_num:
                data[rand_index, 0] = choice(other_label)
                n_index += 1
                noise_index_list.append(rand_index)
            if n_index >= noise_num:
                break
        if flag == 0:
            new_data = data
            flag = 1
        else:
            new_data = vstack([new_data, data])
    return new_data



def load_data(data, noise_rate):
    if noise_rate == 0:
        return data
    train = add_flip_noise(data, noise_rate)
    return train


def main_train_adp(classifier, data, noise):
    train = load_data(data, noise)
    dd, save_best, save_noise, time_Kmeans, tmp_noise, train_kmeans = crfnfl_adp(classifier, train)
    return train, dd, save_best, save_noise, time_Kmeans, tmp_noise, train_kmeans
