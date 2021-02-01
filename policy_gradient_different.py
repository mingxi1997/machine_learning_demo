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
        # self.fc2= nn.Linear(36, 36)
        self.fc3= nn.Linear(36, 2)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)        
    def forward(self,x):
        out=torch.relu(self.fc1(x))
        # out=self.fc2(out)
        out=self.fc3(out)
        
        
        out=torch.nn.Softmax(dim=0)(out)
        return out
   

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


     
model=NN()       
        

env = gym.make('CartPole-v0')
env._max_episode_steps = 500
status = env.reset()

count=0

action_set=[]
reward_set=[]
status_set=[]
discount=0.9

print('collecting data')
for i in tqdm(range(100)):
    status = env.reset()  
    raw_reward_set=[]
    done=False
    while not done:
        status_set.append(status)
    
    
    
        count+=1
        # env.render()
        action=strategy_raw(status)
    
        action_set.append(action)
        status,reward,done,_=env.step(action)
    
    # if done:
    #     reward=0
    
        raw_reward_set.append(reward)
    
    # print(model(torch.tensor(status).to(torch.float32)))
    raw_reward_set=gen_reward(raw_reward_set)
    reward_set.extend(raw_reward_set)
device=torch.device('cuda:0')

actions=torch.tensor(action_set).to(torch.float32).to(device)

rewards=np.array(reward_set)
# rewards=rewards-sum(rewards)/len(rewards)

reward_mean = np.mean(rewards)
reward_std = np.std(rewards)
rewards= (rewards - reward_mean) / reward_std



rewards=torch.tensor(rewards).to(torch.float32).to(device)

statuses=torch.tensor(status_set)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


model=model.to(device)

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

       
        # action=strategy_raw(status)
    
        status,reward,done,_=env.step(action)
        
  print('count',count)
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



num_epochs=200

for epoch in range(num_epochs):
  test_count()
  model.train()
  for i in range(len(actions)):
    optimizer.zero_grad()
    
    
    prob=model(statuses[i].to(torch.float32).to(device))
    
    m=torch.distributions.Categorical(prob)
   
   
    loss=-torch.pow(np.e,m.log_prob(actions[i]))*rewards[i]/0.5
    
    # loss=m.log_prob(actions[i])*rewards[i]
    
    mloss.add(loss.item())
    loss.backward()
    optimizer.step()





