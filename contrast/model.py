import torch
import torch.nn as nn
# three layer MLP
class MLP(nn.Module):
    # define nn
    def __init__(self,input_num,output_num):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_num, int(2*(input_num+output_num)/3))
        self.fc2 = nn.Linear(int(2*(input_num+output_num)/3),output_num)
        self.m=nn.ReLU()

    def forward(self, X):
        X=self.m(self.fc1(X))
        X = self.fc2(X)
        return X
