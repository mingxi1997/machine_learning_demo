import gym
import random
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm



def strategy_raw(status):
    action=random.choice((0,1))
    return action


class NN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1= nn.Linear(4, 36)
        self.fc2= nn.Linear(36, 2)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    def forward(self,x):
        out=self.fc1(x)
        out=torch.relu(out)
        out=self.fc2(out)
        out=torch.nn.Softmax(dim=0)(out)
        return out

    


class AA(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1= nn.Linear(4, 36)
 
        
        self.critic=nn.Linear(36,1)
               
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                
                
    def forward(self,x):
        out=self.fc1(x)
       
       
        
        c=self.critic(torch.tanh(out))
     
        return c
   



def accumulate(s,reward):
    result=0
    for i in range(0,len(reward)-s):

        result+=reward[i]*pow(0.9,i)
    return result

def gen_reward(reward):  
     n_reward=[]
     for i in range(len(reward)):
         n_reward.append(accumulate(i,reward))
     return n_reward


device=torch.device('cuda:0')
     
model=NN().to(device)

cmodel=AA().to(device)
env = gym.make('CartPole-v0')
env._max_episode_steps = 500
status = env.reset()

count=0




discount=0.9

class loss_set:
    def __init__(self):
        self.sum=0
        self.n=0
    def add(self,num):
        self.sum+=num
        self.n+=1
    def show(self):
        out=self.sum/self.n
        self.sum=0
        self.n=0
        return out
    
mloss=loss_set()



def strategy_nn(status):
    x=torch.tensor(status).to(torch.float32).to(device)
    y=model(x)
        
    a=torch.distributions.Categorical(y)
        
    action=int(a.sample().item())
    return action

def test_count():
  model.eval()


  status = env.reset()  
    
  done=False
  count=0
  while not done:
        count+=1
        #env.render()
        
        x=torch.tensor(status).to(torch.float32).to(device)
        y=model(x)
        m=torch.distributions.Categorical(y)
        
        
        action=int(m.sample().item())

        status,reward,done,_=env.step(action)
        
  print('count',count)    
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
coptimizer = torch.optim.Adam(cmodel.parameters(), lr=0.001)


for s in range(10000):
    test_count()

    model.train()
    
    experiences=[]
    
    for i in range(1):
        done=False
        status = env.reset()  
    
        while not done:
            
            experience=[]
            experience.append(status)
  
            action=strategy_nn(status)
    
            experience.append(action)
            status,reward,done,_=env.step(action)

            experience.append(done)

            experience.append(reward)
            experiences.append(experience)
   



    num_epochs=1
 

    for epoch in range(num_epochs):
        I=1
        for i in range(len(experiences)-1):
            coptimizer.zero_grad()
            
            status=torch.tensor(experiences[i][0]).to(torch.float32).to(device)
            
            next_status=torch.tensor(experiences[i+1][0]).to(torch.float32).to(device)
            
            done=torch.tensor(experiences[i+1][2]).to(torch.float32).to(device)
            
            action=torch.tensor(experiences[i][1]).to(torch.float32).to(device)
            
            s_value=cmodel(status)[0]
    
            n_s_value=cmodel(next_status)[0]
            
            target_value=1+(1-done)*0.9*n_s_value
    
    
            closs=torch.nn.MSELoss()(s_value,target_value.detach())
            
            
            
            # print(closs.item())
            
            closs.backward()
            
            coptimizer.step()
      
            optimizer.zero_grad()

            prob=model(status)
    
            a=torch.distributions.Categorical(prob)
    
    
            loss=-a.log_prob(action)*(target_value-s_value).item()*I
            mloss.add(loss.item())
            loss.backward()
            optimizer.step()
            I*=0.9
         




















