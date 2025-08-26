import sys
import numpy as np
import torch.utils.data as Data
# Read the data from corresponding folder
def read_the_data(prefix_name,i):

    attr=[]
    file_object=open(prefix_name+"_"+str(i)+'_'+"train_attr.txt",'r')
    for line in file_object:
        line=line.strip('\n').split(' ')
        attr.append([float(j) for j in line])
    

    train_noisy_label=[]
    file_object=open(prefix_name+"_"+str(i)+'_'+"train_noisy_label.txt",'r')
    for  line in file_object:
        line=line.strip('\n')
        train_noisy_label.append(int(line))
    
    
    test_attr=[]
    file_object=open(prefix_name+'_'+str(i)+'_'+"test_attr.txt",'r')
    for line in file_object:
        line=line.strip('\n').split(' ')
        test_attr.append([float(j) for j in line])
    
    
    test_label=[]
    file_object=open(prefix_name+'_'+str(i)+'_'+"test_label.txt",'r')
    for line in file_object:
        line=line.strip('\n')
        test_label.append(int(line))
    return attr,train_noisy_label,test_attr,test_label


def get_both_pi(data_name,i):
    file_name=data_name+'_'+str(i)+'_pi.txt'
    file_object=open(file_name)
    return float(next(file_object)),float(next(file_object))
    

