import gym
import random
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from collections import namedtuple
import random

class NN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1= nn.Linear(4, 128)
        self.fc2= nn.Linear(128, 2)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  
    def forward(self,x):
        out=self.fc1(x)
        out=torch.relu(out)
        out=self.fc2(out)
        
        return out
global show
device=torch.device('cuda:0')
model=NN().to(device)
env = gym.make('CartPole-v0')
env._max_episode_steps = 2000
SA=namedtuple('SA',field_names=['status','action'])

RSA=namedtuple('RSA',field_names=['rewards','sa_set'])
def collect_data(env,model):
    rsa_set=[]
    for i in range(16):
        status = env.reset()  

        done=False
        rewards=0
        sa_set=[]
        while not done:
        # env.render()
            model.eval()
            x=torch.tensor(status).to(torch.float32).to(device)
            y=model(x)
            y=nn.Softmax(dim=-1)(y)

            a=torch.distributions.Categorical(y)
            action=int(a.sample().item())
            
            sa_set.append(SA._make([status,action]))
            status,reward,done,_=env.step(action)
            rewards+=reward
        rsa_set.append(RSA._make([rewards,sa_set]))
    return rsa_set
        

def rsa_filter(rsa_set):
    rewards=np.zeros(16)
    filted_rsa_set=[]
    for i in range(len(rsa_set)):
        rewards[i]=rsa_set[i].rewards
        
    print(rewards.mean())
    global show
    show=rewards

    for i in range(len(rsa_set)):
        if rsa_set[i].rewards>np.percentile(rewards,70):
            filted_rsa_set.extend(rsa_set[i].sa_set)
    return filted_rsa_set

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion=nn.CrossEntropyLoss()
for _ in range(1000):
    rsa_set= collect_data(env,model)
    filted_rsa_set= rsa_filter(rsa_set)

    s=torch.tensor([n.status for n in filted_rsa_set]).to(torch.float32).to(device)
    a=torch.tensor([n.action for n in filted_rsa_set]).to(device)
    
    
    # train_set=random.sample(filted_rsa_set,128)    
    # for sa in filted_rsa_set:
    
    optimizer.zero_grad()
    
    y=model(s)
        
        
    loss=criterion(y,a)
    loss.backward()
    optimizer.step()
        # print(loss.item())
            
    
    # print(rsa_set[i].rewards)















