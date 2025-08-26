
import time
import torch.nn.functional as F

import numpy as np
import torch

from contrast.Multi_class_CWD.utils.core import accuracy
# from utils.utils import *
from contrast.Multi_class_CWD.utils.meter import AverageMeter
import copy
from contrast.model import MLP
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score


inverse_M_list=[]
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
lr = 0.005
epochs = 1000

def adjust_learning_rate( optimizer, epoch, lr_plan, beta1_plan):
    for param_group in optimizer.param_groups:
        param_group['lr']=lr_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1

def get_assumed_pi_matrix(clean_pi,flip_matrix):
    # The shape of assumed_pi_matrix is (clean_pi,clean_pi).
    assumed_pi_matrix=np.zeros(len(clean_pi)*len(clean_pi)).reshape(len(clean_pi),len(clean_pi))

    for c in range(len(clean_pi)):
        # Set the c-th element of temp_pi as 0.
        temp_pi=copy.deepcopy(clean_pi)
        temp_pi[c]=0
        for j in range(len(clean_pi)):
            if c==j:
                # When j is equal to c.
                assumed_pi_matrix[c][j]=clean_pi[c]+temp_pi@flip_matrix[:,j]
            else:
                # When j is not equal to c.
                assumed_pi_matrix[c][j]=temp_pi@flip_matrix[:,j]
    return assumed_pi_matrix


def generate_noise_transition_matrix(n, noise_level=0.2):
    # 对角线上的概率（标签不变的概率）
    p_stay = 1 - noise_level

    # 初始化转移矩阵
    transition_matrix = np.full((n, n), noise_level / (n - 1))  # 非对角线元素初始化为均匀的转移概率
    np.fill_diagonal(transition_matrix, p_stay)  # 对角线上的元素是标签保持不变的概率

    # 对称矩阵：确保矩阵是对称的
    transition_matrix = (transition_matrix + transition_matrix.T) / 2

    # 确保每行和每列的和为1（进行归一化）
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix /= row_sums  # 按行归一化

    return transition_matrix

def get_auxiliary_transition_matrix(clean_pi,flip_matrix):

    auxiliary_transition_matrix=[]
    for c in range(len(clean_pi)):
        # Initialize this matrix as an identity matrix, and we focus on the c-th row.
        temp_transition_matrix=np.eye((len(clean_pi)))

        for j in range(len(clean_pi)):
            # Similar to get_assumed_pi_matrix() function, discuss the value of j and c separately.
            temp_clean_pi=copy.deepcopy(clean_pi)
            temp_clean_pi[c]=0
            denominator=clean_pi[c]+temp_clean_pi@flip_matrix[:,c]  #Calculate the denominator.
            # Calculate the Eq.(24) and Eq.(26).
            if j!=c: # When j not equal to c.
                numerator=clean_pi[c]*flip_matrix[c][j]
                temp_transition_matrix[c][j]=numerator/denominator
            else:    # When j equal to c.
                numerator=clean_pi[c]*(sum(flip_matrix[c,:])-flip_matrix[c][c])
                temp_transition_matrix[c][c]=1-numerator/denominator
        auxiliary_transition_matrix.append(temp_transition_matrix)
    return auxiliary_transition_matrix

def get_inverse_matrix(assumed_pi_matrix,auxiliary_transition_matrix,clean_pi):
    global inverse_M_list
    ####### Calculate the Eq.(29).
    for c in range(len(clean_pi)):
        c_inverse_matrix=np.zeros((len(clean_pi),len(clean_pi)))
        for j in range(len(clean_pi)):
            temp_inverse_matrix=np.zeros((len(clean_pi),len(clean_pi)))
            for k in range(len(clean_pi)):
                K=np.identity(len(clean_pi)) # The switch matirx is an identity matrix.
                K[[j,k], :]=K[[k,j], :] # Swap the row i and row j in the matrix.
                temp_inverse_matrix+=auxiliary_transition_matrix[c][j][k]*K
            # Get the M-th matrix.
            c_inverse_matrix+=assumed_pi_matrix[c][j]*temp_inverse_matrix
            # Invert it and put it in inverse_M_list.
        inverse_M_list.append(torch.tensor(np.linalg.inv(c_inverse_matrix))) 
    ########### The Eq.(29) is done.
    
    return inverse_M_list


def my_loss(n_classes, outputs, label):
    global inverse_M_list
    label = torch.zeros((outputs.size(0), n_classes)).to(device).scatter_(1, label.view(-1, 1), 1)
    cur_inverse=None
    # Calculate the Eq.(30).
    for i in range(n_classes):
        if i==0:
            cur_inverse=inverse_M_list[i].float()
        else:  
            cur_inverse+=inverse_M_list[i].float()
    cur_inverse=cur_inverse.to(device)
    estimated_inverse = cur_inverse - (n_classes - 1) * torch.eye(n_classes).to(device)
    noisy_centroid = 1 / outputs.size(0) * (outputs.T @ label)
    loss = 1 + (outputs ** 2).sum(1).mean() - 2 * torch.trace(noisy_centroid @ estimated_inverse)
    
    return loss

def train(feature,labels,model, optimizer,n_classes, logger=None):
    
    train_total=0
    train_correct=0 
    loss_meter = AverageMeter()


    # Forward + Backward + Optimize
    logits1=model(feature)
    acc = accuracy(logits1, labels, topk=(1, ))[0]
    train_total+=1
    train_correct+=acc
    loss = my_loss(n_classes, logits1, labels)
    loss_meter.update(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_acc=float(train_correct)/float(train_total)
    return train_acc

# Evaluate the Model using mean prediction of the two networks
def evaluate(fearure,labels,  model, logger=None):
    model.eval()    # Change model to 'eval' mode.
    correct = 0
    total = 0
    images = fearure
    label = labels
    logits = model(images)
    label_distribution = F.softmax(logits, dim=1)
    _, pred = torch.max(label_distribution.data, 1)
    pred = pred.detach().cpu().numpy()
    return accuracy_score(label.data.cpu().numpy(), pred), precision_score(label.data.cpu().numpy(), pred, average='macro'), recall_score(
        label.data.cpu().numpy(), pred, average='macro'), f1_score(label.data.cpu().numpy(), pred, average='macro')


def run(train_ds,t_labels_num,eval_ds,eval_label,config_wy,train_ds_,eval_ds_,meta):
    best_accuracy = 0
    # model, optimizer, scheduler --------------------------------------------------------------------------------------------------------------------
    n_classes = config_wy.data.output_size
    #labels = F.one_hot(t_labels_num, num_classes=n_classes).to(device)
    labels = t_labels_num
    #eval_label = F.one_hot(eval_label, num_classes=n_classes).to(device)
    cnn = MLP(train_ds.shape[1],n_classes).to(device)

    # Adjust learning rate and betas for Adam Optimizer
    epoch_decay_start = 80
    mom1 = 0.9
    mom2 = 0.1
    lr_plan = [lr] * epochs
    beta1_plan = [mom1] * epochs
    for i in range(epoch_decay_start, epochs):
        lr_plan[i] = float(epochs - i) / (epochs - epoch_decay_start) * lr
        beta1_plan[i] = mom2



    
    optimizer = torch.optim.Adam(cnn.parameters(), lr)
    
    # meters -----------------------------------------------------------------------------------------------------------------------------------------
    train_loss = AverageMeter()
    
    flip_matrix = generate_noise_transition_matrix(n_classes, config_wy.training.noise)

    # the noisy_pi in Animal-10N is all 0.1 
    noisy_pi=np.array([1 / n_classes for i in range(n_classes)])

    # Get the clean_pi via Eq.(29) ---> Multi-class CWD step 3.
    clean_pi=np.linalg.solve(flip_matrix.T,noisy_pi)

    # Get the pi_{\tilde_c,j} via Eq.(21) and Eq.(22) ----> Multi-class CWD step 5.
    # The size of this assumed_pi_matrix is cfg.n_classes*cfg.n_classes
    assumed_pi_matrix=get_assumed_pi_matrix(clean_pi,flip_matrix)

    # Get the flip matrix in Eq.(28) via Eqs.(23), Eq.(24), Eq.(25), Eq.(26). ----> Multi-class CWD step 6.
    # The size of auxiliary_transition_matrix is also cfg.n_classes*cfg.n_classes.
    auxiliary_transition_matrix=get_auxiliary_transition_matrix(clean_pi,flip_matrix)
    
    # Get the inverse matrix list ---> Multi-class CWD step 7.
    # The size of inverse_M_list is also cfg.n_classes.
    inverse_M_list=get_inverse_matrix(assumed_pi_matrix,auxiliary_transition_matrix,clean_pi)
    
    # training ---------------------------------------------------------------------------------------------------------------------------------------
    for epoch in range(epochs):
        start_time = time.time()

        # pre-step in this epoch
        train_loss.reset()
        curr_lr = [group['lr'] for group in optimizer.param_groups]

        
        
        cnn.train()
        adjust_learning_rate(optimizer, epoch, lr_plan, beta1_plan)
        
        train(train_ds,labels, cnn, optimizer,n_classes)
        
        # evaluate models
        test_acc=evaluate(eval_ds, eval_label, cnn)[0]

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_epoch = epoch + 1


    return cnn


        



