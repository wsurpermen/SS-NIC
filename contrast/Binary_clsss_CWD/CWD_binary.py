import numpy as np
import torch 
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from torch.autograd import Variable

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

# parser = argparse.ArgumentParser()
# parser.add_argument('--learning_rate', type = float, default = 0.001)
# parser.add_argument('--epoch', type=int, default=200)
# parser.add_argument('--seed', type=int, default=1)
# parser.add_argument('--pos_noise_rate', type=float, default=0.2)
# parser.add_argument('--neg_noise_rate', type=float, default=0.2)
# parser.add_argument('--model_type', type = str, default='CWD binary classification model')
# args = parser.parse_args()
class Dict:
    def __init__(self):
        self.learning_rate = 0.005
        self.seed = 1
        self.epoch = 1000



args = Dict()



# Define some hyperparameter
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epoch=args.epoch
learning_rate=args.learning_rate

# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)



def my_loss(ret,target,estimated_centroid_coefficient): #Calculate the loss via Eq.(4)
    ret=torch.mean(ret.squeeze()**2+1-2*estimated_centroid_coefficient*ret.squeeze()*target)
    return ret

def train(model, optimizer, train_attr, train_label, estimated_centroid_coefficient): # Train the model
    
    model.train()
    global epoch
    for _ in range(epoch):
        optimizer.zero_grad()
        out = model(train_attr)
        loss = my_loss(out, train_label,estimated_centroid_coefficient)
        loss.backward()
        optimizer.step()
    
    
def evaluate(test_attr,label,model): # Evaluate our model performance
    model.eval()
    predict_out = model(test_attr).cpu().detach().numpy()
    index=predict_out>0.5
    predict_out[index]=1.0
    predict_out[~index]=0.0
    return accuracy_score(label.data.cpu().numpy(),predict_out),precision_score(label.data.cpu().numpy(),predict_out),recall_score(label.data.cpu().numpy(),predict_out),f1_score(label.data.cpu().numpy(),predict_out,average='binary')


def calculate_centroid_coefficient(pos_noise_rate,neg_noise_rate,pos_pi,neg_pi): # Calculate the centorid coefficient via Eq.(15) 
    eta_positive=pos_noise_rate
    eta_negative=neg_noise_rate
    # Just calculate the coefficient of S
    first_item=1/(1-2*pos_pi*eta_positive)
    second_item=1/(1-2*neg_pi*eta_negative)
    return first_item+second_item-1

def run(train_attr, train_noisy_label, noise_rate):
    pos_noise_rate=float(noise_rate) # The positive noise rate
    neg_noise_rate=float(noise_rate)
    # CWD-binary algorithm step 2. Calculate the Pi_P and Pi_N
    pos_pi = (np.sum(np.array(train_noisy_label.detach().cpu()) == 1) / len(train_noisy_label) - neg_noise_rate) / (
                1 - pos_noise_rate - neg_noise_rate)
    neg_pi = 1 - pos_pi

    ##### Define the input and output dimension here
    input_num = len(train_attr[0])
    output = 1
    #####
    model = MLP(input_num, output)  # Define the backbone model

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    estimated_centroid_coefficient = calculate_centroid_coefficient(pos_noise_rate, neg_noise_rate, pos_pi,
                                                                    neg_pi)  # Calculate the centroid coefficient

    train(model, optimizer, train_attr, train_noisy_label, estimated_centroid_coefficient)



    return model



if __name__=='__main__':
    from contrast.Binary_clsss_CWD.read_data import read_the_data
    pos_noise_rate=float(args.pos_noise_rate) # The positive noise rate
    neg_noise_rate=float(args.neg_noise_rate) # The negative noise rate
    data_name='GermanCredit'
    prefix_name=data_name+'/'+str(pos_noise_rate)+'_'+str(neg_noise_rate)
    # last_acc=[]
    # last_mean_acc=-1
    # last_std=-1
    acc_list=[]
    for i in range(5):  # Five-fold validation
        train_attr,train_noisy_label,test_attr,test_label=read_the_data(prefix_name,i) #load the data
        
        # CWD-binary algorithm step 2. Calculate the Pi_P and Pi_N
        pos_pi=(np.sum(np.array(train_noisy_label)==1)/len(train_noisy_label)-neg_noise_rate)/(1-pos_noise_rate-neg_noise_rate)
        neg_pi=1-pos_pi
        #

        ##### Define the input and output dimension here
        input_num=len(train_attr[0])
        output=1
        #####

        train_attr,train_noisy_label,test_attr,test_label=torch.tensor(train_attr).float(),torch.tensor(train_noisy_label).float(),torch.tensor(test_attr).float(),torch.tensor(test_label).float()
        train_attr,test_attr=Variable(train_attr).to(device),Variable(test_attr).to(device)
        train_noisy_label,test_label=Variable(train_noisy_label).to(device),Variable(test_label).to(device)
        
        model=MLP(input_num,output) # Define the backbone model
        
        model.to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

        estimated_centroid_coefficient=calculate_centroid_coefficient(pos_noise_rate,neg_noise_rate,pos_pi,neg_pi) # Calculate the centroid coefficient

        train(model,optimizer,train_attr,train_noisy_label,estimated_centroid_coefficient)
        acc=evaluate(test_attr,test_label,model)
        acc_list.append(acc)
    mean_acc=np.mean(np.array(acc_list))
    std=np.std(np.array(acc_list))
    
        

    print(acc_list)
    print(mean_acc)
    print(std)


    file_name='result.txt'
    file_object=open(file_name,'a')
    file_object.write(data_name+str(pos_noise_rate)+'_'+str(neg_noise_rate)+'\n')
    for i in acc_list:
        file_object.write(str(i)+' ')
    file_object.write('\n')
    file_object.write("last_mean_acc: "+str(mean_acc)+" "+'\n')
    file_object.write("last_std: "+str(std)+" "+'\n')
    file_object.write('\n')
   