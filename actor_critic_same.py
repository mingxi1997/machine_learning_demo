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
   
class actor_critic(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1= nn.Linear(4, 36)
        # self.fc2= nn.Linear(36, 36)
        self.actor= nn.Linear(36, 2)
        self.critic=nn.Linear(36,2)
        
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                
                
    def forward(self,x):
        out=self.fc1(x)
      
        
        a=self.actor(torch.relu(out))
        a=torch.nn.Softmax(dim=0)(a)
        
        c=self.critic(torch.tanh(out))
  
      
        return a,c
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
     
model=actor_critic().to(device)
env = gym.make('CartPole-v0')
env._max_episode_steps = 500
status = env.reset()

count=0

action_set=[]
reward_set=[]
status_set=[]
discount=1

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
    a,c=model(x)
        
    a=torch.distributions.Categorical(a)
        
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
        a,c=model(x)
        m=torch.distributions.Categorical(a)
        
        
        action=int(m.sample().item())

        status,reward,done,_=env.step(action)
        
  print('count',count)    

discount=0.95
for s in range(100):
 test_count()


 for i in range(1):
    status = env.reset()  
    raw_reward_set=[]
    done=False
    while not done:
        status_set.append(status)
    
    
    
        count+=1
        # env.render()
        action=strategy_nn(status)
    
        action_set.append(action)
        status,reward,done,_=env.step(action)
    
    # if done:
    #     reward=0
    
        raw_reward_set.append(reward)
    
    # print(model(torch.tensor(status).to(torch.float32)))
    raw_reward_set=gen_reward(raw_reward_set)
    reward_set.extend(raw_reward_set)

 actions=torch.tensor(action_set).to(torch.float32).to(device)

 rewards=np.array(reward_set)
# rewards=rewards-sum(rewards)/len(rewards)

 reward_mean = np.mean(rewards)
 reward_std = np.std(rewards)
 rewards= (rewards - reward_mean) / reward_std



 rewards=torch.tensor(rewards).to(torch.float32).to(device)

 statuses=torch.tensor(status_set)

 optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



 num_epochs=1

 for epoch in range(num_epochs):
  discount=1
  for i in range(len(actions)-1):
    discount*=0.95
    optimizer.zero_grad()
    a,c=model(statuses[i].to(torch.float32).to(device))
    
    na,nc=model(statuses[i+1].to(torch.float32).to(device))
    
    pi_a=torch.distributions.Categorical(a).sample()
    pi_na=torch.distributions.Categorical(a).sample()
    
    
    predict_q=nc[pi_na.item()]*0.95+1
    predict_q=predict_q.detach()
    
    
    aloss=-torch.distributions.Categorical(a).log_prob(actions[i])*predict_q*discount
    
    closs=torch.nn.MSELoss()(predict_q,c[pi_a.item()])
    
    loss=aloss+closs
    # mloss.add(loss.item())
    loss.backward()
    optimizer.step()
