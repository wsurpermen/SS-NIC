import torch
import torch.nn as nn
from utils.utils import init_weights
import torch.nn.functional as F

# non-linear projection head
class NoneLinearProjectionHead(nn.Module):
    def __init__(self, dim_in=2048, dim_out=128, dim_hidden=2048):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_hidden)
        self.bn1 = nn.BatchNorm1d(dim_hidden)
        self.relu1 = nn.ReLU(True)
        self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        self.bn2 = nn.BatchNorm1d(dim_hidden)
        self.relu2 = nn.ReLU(True)
        self.linear3 = nn.Linear(dim_hidden, dim_out)
        
    def forward(self, x):
        x = self.linear1(x).unsqueeze(-1).unsqueeze(-1)
        x = self.bn1(x).squeeze(-1).squeeze(-1)
        x = self.relu1(x)
        x = self.linear2(x).unsqueeze(-1).unsqueeze(-1)
        x = self.bn2(x).squeeze(-1).squeeze(-1)
        x = self.relu2(x)
        x = self.linear3(x)
        return x
    
class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden, projection_size, init_method='He', with_BN=True, projection_dim = None):
        super().__init__()
        if projection_dim is None:
            mlp_hidden_size = round(mlp_hidden * in_channels)
        else:
            mlp_hidden_size = projection_dim
        if with_BN:
            self.mlp_head = nn.Sequential(
                nn.Linear(in_channels, mlp_hidden_size),
                nn.BatchNorm1d(mlp_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden_size, projection_size)
            )
        else:
            self.mlp_head = nn.Sequential(
                nn.Linear(in_channels, mlp_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden_size, projection_size)
            )
        init_weights(self.mlp_head, init_method)

    def forward(self, x, projection_ret = False):
        if not projection_ret:
            return self.mlp_head(x)
        else:
            projection = self.mlp_head[:-1](x)
            logits = self.mlp_head[-1](projection)
            return projection, logits