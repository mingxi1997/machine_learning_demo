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
                               
                                 )
        
        self.actor=nn.Linear(128,action_nums)
        self.critic=nn.Linear(128,1)
               
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                
                
    def forward(self,x):
        out=self.fc(x)
       
        a=torch.nn.Softmax(dim=-1)(self.actor(torch.relu(out)))
        c=self.critic(torch.relu(out))
     
        return a,c
    

device=torch.device('cuda:0')
status_nums=4
action_nums=2
model=AC().to(device)

env = gym.make('CartPole-v0')
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
        policy,value=model(x)
       
        dist=torch.distributions.Categorical(policy)
        
            
        action=int(dist.sample().item())
        
    
        return action,value,policy
 



for s in range(10000):
    # epsilon_clip*=0.99
    exp=[]
    
    
    done=False
    status = env.reset()  
    c=0
    while not done:
        c+=1
        
        
        experience=[]
        experience.append(status)
  
        action,value,policy=choose_action(status)
        
        action_one_hot = torch.zeros(2)
        action_one_hot[action] = 1

        experience.append(action_one_hot)
      
        status,reward,done,_=env.step(action)

        if done:
            reward=-1.
            
            

        experience.append(reward)
        experience.append(policy)
        experience.append(value)

        exp.append(experience)
    
                
        
    print('step :{} score:{}'.format(s,c))
    score=c
    running_score = 0.99 * running_score + 0.01 * score
    print(running_score)
    writer.add_scalar('c',c,s)
    writer.add_scalar('running_score',running_score,s)
  
    nexp=list(zip(*exp))
   
    status_set=torch.tensor(nexp[0]).to(torch.float32).to(device)
    action_set=torch.stack(nexp[1]).to(torch.float32).to(device)
    reward_set=nexp[2]
    old_policies=torch.stack(nexp[3]).to(torch.float32).to(device)
    old_values=torch.stack(nexp[4]).to(torch.float32).to(device)
    
   
    returns,advantages   =get_advantage(torch.tensor(reward_set), old_values.cpu())
    
    advantages=advantages.to(torch.float32).to(device).detach()
    returns=returns.to(torch.float32).to(device).detach()



    for _ in range(ppo_epoch):
  
        # model.train()

        npolicy,nvalues=model(status_set)
    
        critic_loss = (returns - nvalues.squeeze()).pow(2).mean()
    
    
        ratios = ((npolicy / old_policies) * action_set).sum(dim=1)
        clipped_ratios = torch.clamp(ratios, min=1.0-epsilon_clip, max=1.0+epsilon_clip).squeeze(0)
        actor_loss = -torch.min(ratios*advantages ,clipped_ratios*advantages ).mean()
        policy_entropy = (torch.log(npolicy) * npolicy).sum(1, keepdim=True).mean()
        loss = actor_loss + 0.5*critic_loss - 0.01* policy_entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
   
    


    
         
