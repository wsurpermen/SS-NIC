import numpy as np
from collections import Counter
from random import choice
import warnings
import time
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

minNumSample = 80


class BinaryTree:
    def __init__(self, labels=np.array([]), datas=np.array([])):
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
    m, n = data.shape
    data = np.hstack((data, noiseOrder.reshape(m, 1)))
    redata = np.array(list(filter(lambda x: x[n] == 0, data[:, ])))
    if len(redata) == 0:
        return redata
    else:
        redata = np.delete(redata, n, axis=1)
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
            labels = np.zeros(len(data)) + labels
        except TypeError:
            pass
        result = np.vstack((data, labels))
        return result
    else:
        resultLeft = visitCRT(tree.get_leftChild())
        resultRight = visitCRT(tree.get_rightChild())
        result = np.hstack((resultLeft, resultRight))
        return result



def splitData(data, splitAttribute, splitValue):
    rightData = np.array(list(filter(lambda x: x[splitAttribute] > splitValue, data[:, ])))
    leftData = np.array(list(filter(lambda x: x[splitAttribute] <= splitValue, data[:, ])))
    return leftData, rightData


def generateTree(data, uplabels=[]):
    try:
        numberSample, numberAttribute = data.shape
    except ValueError:
        numberSample = 1
        numberAttribute = data.size

    if numberAttribute == 0:
        return None

    numberAttribute = numberAttribute - 2  # for index, eg: a = [1,2], a[2-1] = 2
    labelNumKey = []
    if numberSample == 1:
        labelvalue = data[0][0]
        rootdata = data[0][numberAttribute + 1]   # rootdata is the order
    else:
        labelNum = Counter(data[:, 0])
        labelNumKey = list(labelNum.keys())
        labelNumValue = list(labelNum.values())
        labelvalue = labelNumKey[labelNumValue.index(max(labelNumValue))]
        rootdata = data[:, numberAttribute + 1]
    rootlabel = np.hstack((labelvalue, uplabels))

    CRTree = BinaryTree(rootlabel, rootdata)

    if numberSample < minNumSample or len(labelNumKey) < 2:  # minNumSample minNumSample The default is 10 or there is only 1 type of tag lef
        return CRTree
    else:
        maxCycles = 1.5 * numberAttribute
        i = 0
        while True:
            i += 1
            splitAttribute = np.random.randint(1, numberAttribute)
            if splitAttribute > 0 and splitAttribute < numberAttribute + 1:
                dataSplit = data[:, splitAttribute]
                uniquedata = list(set(dataSplit))
                if len(uniquedata) > 1:
                    break
            if i > maxCycles:
                return CRTree
        sv1 = np.random.choice(uniquedata)
        i = 0
        while True:
            i += 1
            sv2 = np.random.choice(uniquedata)
            if sv2 != sv1:
                break
            if i > maxCycles:
                return CRTree
        splitValue = np.mean([sv1, sv2])
        leftdata, rightdata = splitData(data, splitAttribute, splitValue)
        CRTree.set_leftChild(generateTree(leftdata, rootlabel))
        CRTree.set_rightChild(generateTree(rightdata, rootlabel))
        return CRTree


# Add the ordinal column at the end and call the function generateTree spanning tree
def CRT(data):
    numberSample, numberAttribute = data.shape
    orderAttribute = np.arange(numberSample).reshape(numberSample, 1)  # (646, 1)
    data = np.hstack((data, orderAttribute))  #  Add another column at the end
    completeRandomTree = generateTree(data)
    return completeRandomTree


def Compare(save_best, best_parament):
    if save_best == []:
        save_best = best_parament.copy()
    else:
        if save_best[2] < best_parament[2]:
            save_best = best_parament.copy()
        elif save_best[2] == best_parament[2]:
            if save_best[1] > best_parament[1]:
                save_best = best_parament.copy()
            elif save_best[1] == best_parament[1]:
                if save_best[0] > best_parament[0]:
                    save_best = best_parament.copy()
    return save_best


def Compare_and_noise(noise_data, noise_data_return, save_best, best_parament):
    if save_best == []:
        save_best = best_parament.copy()
        noise_data_return = noise_data
    else:
        if save_best[2] < best_parament[2]:
            save_best = best_parament.copy()
            noise_data_return = noise_data
        elif save_best[2] == best_parament[2]:
            if save_best[1] > best_parament[1]:
                save_best = best_parament.copy()
                noise_data_return = noise_data
            elif save_best[1] == best_parament[1]:
                if save_best[0] > best_parament[0]:
                    save_best = best_parament.copy()
                    noise_data_return = noise_data
    return noise_data_return, save_best


def CRFNFL(classifier, traindata, val, ntree=20, niThreshold=11):
    best_parament = [0, 0, 0, 0]
    save_best = []
    if ntree < 1:
        print('The value of ntree at least is 1.')
        return 0

    try:
        m, n = traindata.shape  #  m column ,n line
    except ValueError as e:
        print(str(e))
        return 0

    forest = np.array([])  #matrix: m column,ntree line
    # time_start = time.clock()
    for i in range(ntree):
        tree = CRT(traindata)  # Random spanning tree
        visiTree = visitCRT(tree)
        visiTree = visiTree[:, np.argsort(visiTree[0, :])]
        visiTree = visiTree[1, :]
        if forest.size == 0:
            forest = visiTree.reshape(m, 1)  # m column  , 1 line
        else:
            forest = np.hstack((forest, visiTree.reshape(m, 1)))
    noise_data_return = np.zeros(m)
    if ntree < 10:
        # print('森林规模小于10，只讨论niThreshold的变化情况下的最优精度。')
        for subNi in range(2, niThreshold + 1):
            noiseForest = np.zeros(m)
            for j in range(m):
                for k in range(ntree):
                    if forest[j, k] >= subNi:
                        noiseForest[j] += 1
                if noiseForest[j] >= 0.5 * ntree:  # votes
                    noiseForest[j] = 1
                else:
                    noiseForest[j] = 0
            noise_data = noiseForest
            denoiseTraindata = deleteNoiseData(traindata, noiseForest)
            best_parament[2] = Func(classifier, denoiseTraindata, val)  # Call the system basic classification algorithm
            best_parament[0] = subNi
            best_parament[1] = ntree
            best_parament[3] = sum(noise_data)
            noise_data_return, save_best = Compare_and_noise(noise_data, noise_data_return, save_best, best_parament)


    else:
        startNtree = 1
        endNtree = ntree // 10
        remainderNtree = ntree % 10
        for i in range(startNtree, endNtree + 1):
            subNtree = i * 10
            for subNi in range(2, niThreshold + 1):
                noiseForest = np.zeros(m)
                for j in range(m):
                    for k in range(subNtree):
                        if forest[j, k] >= subNi:
                            noiseForest[j] += 1
                    if noiseForest[j] >= 0.5 * subNtree:
                        noiseForest[j] = 1
                    else:
                        noiseForest[j] = 0
                noise_data = noiseForest
                denoiseTraindata = deleteNoiseData(traindata, noiseForest)
                try:
                    best_parament[2] = Func(classifier, denoiseTraindata, val)  # Basic classifier classification results
                except:
                    best_parament[2] = 0
                best_parament[0] = subNi
                best_parament[1] = subNtree
                best_parament[3] = sum(noise_data)
                noise_data_return, save_best = Compare_and_noise(noise_data, noise_data_return, save_best,
                                                                 best_parament)

        if remainderNtree > 0:
            for subNi in range(2, niThreshold + 1):
                noiseForest = np.zeros(m)
                for j in range(m):
                    for k in range(ntree):
                        if forest[j, k] >= subNi:
                            noiseForest[j] += 1

                    if noiseForest[j] >= 0.5 * ntree:
                        noiseForest[j] = 1
                    else:
                        noiseForest[j] = 0
                noise_data = noiseForest
                denoiseTraindata = deleteNoiseData(traindata, noiseForest)
                best_parament[2] = Func(classifier, denoiseTraindata, val)
                best_parament[0] = subNi
                best_parament[1] = ntree
                best_parament[3] = sum(noise_data)
                noise_data_return, save_best = Compare_and_noise(noise_data, noise_data_return, save_best,
                                                                 best_parament)
    # time_CRF = time.clock() - time_start
    save_best.append(0)
    return noise_data_return, save_best


def Func(classifier, traindata, testdata):
    traindatalabel = traindata[:, 0]
    traindata = traindata[:, 1:]
    testdatalabel = testdata[:, 0]
    testdata = testdata[:, 1:]
    classifier.fit(traindata, traindatalabel)
    return classifier.score(testdata, testdatalabel)  # Prediction accuracy


def crfnfl_all(classifier, traindata):
    save_best = []
    dic = {}
    for i in range(len(traindata)):
        dic[i] = 0
    folds = KFold(n_splits=10, shuffle=True)
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(traindata)):
        trn = traindata[trn_idx]
        val = traindata[val_idx]
        noise, best_parament = CRFNFL(classifier, trn, val)  # Outgoing noise point

        save_best = Compare(save_best, best_parament)
        for i in range(len(trn_idx)):
            dic[trn_idx[i]] += noise[i]
    a = []
    for i in dic.keys():
        if dic[i] < 5:
            a.append(i)

    a = np.array(a)
    train = traindata[a] if len(a)!=0 else np.array([])

    return a, save_best


def add_flip_noise(dataset, noise_rate):
    label_cat = list(set(dataset[:, 0]))
    new_data = np.array([])
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
            rand_index = int(np.random.uniform(0, n))
            if rand_index in noise_index_list:
                continue
            if n_index < noise_num:
                data[rand_index, 0] = choice(other_label)  # todo
                n_index += 1
                noise_index_list.append(rand_index)
            if n_index >= noise_num:
                break
        if flag == 0:
            new_data = data
            flag = 1
        else:
            new_data = np.vstack([new_data, data])
    return new_data



def load_data(data, noise_rate):

    # To avoid calling [0,0]
    if noise_rate == 0:
        return data
    train = add_flip_noise(data, noise_rate)  # Add flip noise
    return train


def main_train(classifier, data, noise):
    train = load_data(data, noise)
    dd, save_best = crfnfl_all(classifier, train)
    return train, dd, save_best


if __name__ == '__main__':
    main_train()
