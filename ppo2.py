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
        self.fc3= nn.Linear(36, 1)
    def forward(self,x):
        out=self.fc1(x)
        # out=self.fc2(out)
        out=self.fc3(out)
        out=torch.sigmoid(out)
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


device=torch.device('cuda:0')        
     
model=NN().to(device)       

env = gym.make('CartPole-v0')
env._max_episode_steps = 2000
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


def strategy_nn(status):
    model.eval()
    x=torch.tensor(status).to(torch.float32).to(device)
    y=model(x)
    a=torch.distributions.Bernoulli(y)
    action=int(a.sample().item())
    return action
    
def show_count():
    env = gym.make('CartPole-v0')
    env._max_episode_steps = 2000
    status = env.reset()  
    
    done=False
    count=0
    while not done:
        count+=1
        # env.render()
        model.eval()
        x=torch.tensor(status).to(torch.float32).to(device)
        y=model(x)
        
        a=torch.distributions.Bernoulli(y)
        
        action=int(a.sample().item())

        # print(action)
        # action=strategy_raw(status)
    
        status,reward,done,_=env.step(action)
    if count==2000:
        torch.save(model.state_dict(), 'pole_adv.pth')  
        
    print('\n the {} times count :'.format(year),count)
cliprange=0.1
for year in range(20):
  action_set=[]
  reward_set=[]
  status_set=[]
  probs_set=[]
  status = env.reset()

  count=0


  discount=0.9

  model.eval()
  for i in tqdm(range(10)):
    status = env.reset()  
    raw_reward_set=[]
    done=False
    temp_count=0
    while not done:
        temp_count+=1
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
 
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


  # o_probs=a.log_prob(actions.to(device))

  model.eval()
 
  for i in range(len(statuses)):
      prob=model(statuses[i].to(torch.float32).to(device))
      a=torch.distributions.Bernoulli(prob)
      probs_set.append(a.log_prob(actions[i]))




  num_epochs=10
  model.train()
  for epoch in range(num_epochs):
   for i in range(len(statuses)):
    optimizer.zero_grad()
    prob=model(statuses[i].to(torch.float32).to(device))
    # print(prob.item())
    a=torch.distributions.Bernoulli(prob)
    
    
    # ratio = torch.exp(traj_info['log_pi_a'] - o_probs)
    
   
    ratio = torch.exp(a.log_prob(actions[i]) -probs_set[i])
    
  
    rarion_clamp = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) 
    
    ra = torch.min(ratio, rarion_clamp).detach()
    
    # print(ra)
    
    loss=-(a.log_prob(actions[i]))*rewards[i]/ra
    
    
#     print(loss.item())
#     # mloss.add(loss.item())
    loss.backward()
    optimizer.step()
#     # if (i+1) % 200 == 0:
#     #         print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
#     #                 .format(epoch+1, num_epochs, i+1, len(actions), mloss.show()))
#   # return model


# # rewards=torch.tensor(rewards)
  model.eval()

  show_count()





# # # torch.save(model.state_dict(), 'pole_raw.pth')  

