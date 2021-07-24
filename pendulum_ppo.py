import gym
import random
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter



class AC(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc= nn.Sequential(nn.Linear(status_nums, 128),
                                nn.ReLU(),
                                nn.Linear(128, 128),
                                nn.ReLU(),
                                 )
        
        self.actor= nn.Sequential(nn.Linear(128, action_nums),)
        
        
        
        self.critic=nn.Linear(128,1)
               
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                
                
    def forward(self,x):
        out=self.fc(x)
       
        action_mean = torch.tanh(self.actor(out))*2
       
        
        
        
        
        c=self.critic(torch.relu(out))
     
        return action_mean,c
    

device=torch.device('cuda:0')
status_nums=3
action_nums=1
model=AC().to(device)

env = gym.make('Pendulum-v0')
env._max_episode_steps = 2000
status = env.reset()
gamma = 0.99
lambda_gae = 0.96
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
running_score=0

epsilon_clip=0.1
ciritic_coefficient = 0.5
entropy_coefficient = 0.01
ppo_epoch=30
writer=SummaryWriter()
max_grad_norm=0.5

action_std = 0.6 



def get_advantage(rewards,values):
    
    values=values.squeeze()

        
        
    running_tderror=torch.zeros_like(rewards)
       
    for t in reversed(range(len(rewards))):
         if t==len(rewards)-1:
            running_tderror[t]=rewards[t]-values[t]
         else:
             running_tderror[t]=rewards[t]+gamma*values[t+1]-values[t]
             
             
    advantages=    torch.zeros_like(rewards)
       
    for t in reversed(range(len(rewards))):
         if t==len(rewards)-1:
            advantages[t]=running_tderror[t]
         else:
             advantages[t]=running_tderror[t]+(gamma * lambda_gae)*advantages[t+1]
    returns=advantages+values
    
    returns=(returns-returns.mean())/returns.std()
    advantages=(advantages-advantages.mean())/advantages.std()
    return returns,advantages


def choose_action(status):

    with torch.no_grad():
        x=torch.tensor(status).to(torch.float32).to(device)
        
        
        action_mean,value=model(x)
        
        
        
        dist=torch.distributions.Normal(action_mean, action_std)
        
        action=dist.sample()
        
        return action,value,dist.log_prob(action)
 



for s in range(10000):
    if s%2e5==0:
        
       action_std=action_std-0.05 if action_std>0.1 else action_std
    exp=[]
    
    
    done=False
    status = env.reset()  
    c=0
    rewards_count=0
    for _ in range(2000):
        c+=1
        
        
        experience=[]
        experience.append(status)
  
        action,value,policy=choose_action(status)
        
        

        experience.append(action)
      
        status,reward,done,_=env.step(np.clip(action.cpu().numpy(),-2,2))

 
        experience.append(reward)
        rewards_count+=reward
        experience.append(policy)
        experience.append(value)

        exp.append(experience)
    
                
    if s%10==0:    
        print('step :{} score:{}'.format(s,rewards_count))

    
    writer.add_scalar('running_score',rewards_count,s)
  
            
   
    
    nexp=list(zip(*exp))
    
   
    status_set=torch.tensor(nexp[0]).to(torch.float32).to(device)
    action_set=torch.stack(nexp[1]).to(torch.float32).to(device).squeeze()
    reward_set=torch.tensor(nexp[2]).to(torch.float32).to(device)
    old_policies=torch.stack(nexp[3]).to(torch.float32).to(device).squeeze()
    old_values=torch.stack(nexp[4]).to(torch.float32).to(device).squeeze()
    
   
    returns,advantages =get_advantage(reward_set, old_values)
    
  

    for _ in range(ppo_epoch):
  
       

        action_mean,nvalues=model(status_set)
        
        critic_loss = (returns - nvalues.squeeze()).pow(2).mean()
      
        dist=torch.distributions.Normal(action_mean, action_std)
        
       
        npolicies=dist.log_prob(action_set.unsqueeze(1)).squeeze()
    
    
    
        ratios = torch.exp(npolicies-old_policies)
        
        
        clipped_ratios = torch.clamp(ratios, min=1.0-epsilon_clip, max=1.0+epsilon_clip).squeeze(0)
        actor_loss = -torch.min(ratios*advantages ,clipped_ratios*advantages ).mean()
        
        
        policy_entropy = dist.entropy().squeeze().mean()
        loss = actor_loss + 0.5*critic_loss - 0.01* policy_entropy
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
   
    writer.add_scalar('loss',loss.item(),s)


    
         
