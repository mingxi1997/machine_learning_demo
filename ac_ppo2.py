import gym
import random
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter



ciritic_coefficient = 0.5
entropy_coefficient = 0.01
writer=SummaryWriter()



class AC(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc= nn.Sequential(nn.Linear(4, 128),
                                nn.ReLU(),
                                nn.Linear(128, 128),
                               
                                 )
        
        self.actor=nn.Linear(128,2)
        self.critic=nn.Linear(128,1)
               
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                
                
    def forward(self,x):
        out=self.fc(x)
       
       
        a=torch.nn.Softmax(dim=-1)(self.actor(torch.relu(out)))
        c=self.critic(torch.relu(out))
     
        return a,c
def sample(status_set, action_set, returns, advantages, old_policies,old_values):
    
    
    index=random.sample(list(range(len(status_set))),int(len(status_set)*0.5))
    return status_set[index], action_set[index], returns[index], advantages[index], old_policies[index],old_values[index]


device=torch.device('cuda:0')
     
model=AC().to(device)

# cmodel=AA().to(device)
env = gym.make('CartPole-v0')
env._max_episode_steps = 2000
status = env.reset()
gamma = 0.99

lambda_gae = 0.96


def get_advantage(rewards,values):
    returns=torch.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
    
        if t==len(rewards)-1:
            returns[t]=rewards[t]
        else:
            returns[t]=rewards[t]+gamma*returns[t+1]
        
        
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
    return returns,advantages


def choose_action(status):
    global show

    model.eval()
    x=torch.tensor(status).to(torch.float32).to(device)
    A,C=model(x)
   
    a=torch.distributions.Categorical(A)
    
       

        
    action=int(a.sample().item())
    
    
    


    return action,A
 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
running_score=0

epsilon_clip=0.1


for s in range(10000):
    exp=[]
    
    
    done=False
    status = env.reset()  
    c=0
    while not done:
        c+=1
        
        
        experience=[]
        experience.append(status)
  
        action,critic=choose_action(status)
        
        action_one_hot = torch.zeros(2)
        action_one_hot[action] = 1

        experience.append(action_one_hot)
        # experience.append(policy)
        status,reward,done,_=env.step(action)

        if done:
            reward=-1.
            
            
        # experience.append(status)

        experience.append(reward)
        

        exp.append(experience)
    
                
                        

            
        
    print(c)
    score=c
    running_score = 0.99 * running_score + 0.01 * score
    print(running_score)
    writer.add_scalar('c',c,s)
    writer.add_scalar('running_score',running_score,s)
  
            
   
    
    nexp=list(zip(*exp))
    
   
    status_set=torch.tensor(nexp[0]).to(torch.float32).to(device)
    action_set=torch.stack(nexp[1]).to(torch.float32).to(device)
    

    reward_set=nexp[2]
    
   
    model.eval()

      
    old_policies,old_values=model(status_set)
    
    old_policies = old_policies.detach()
    
    
   
    values=old_values.detach()
    
    
    returns,advantages   =get_advantage(torch.tensor(reward_set), old_values.cpu())
    
    advantages=advantages.to(torch.float32).to(device).detach()
    returns=returns.to(torch.float32).to(device).detach()

    status_set, action_set, returns, advantages, old_policies,old_values=sample(status_set, action_set, returns, advantages, old_policies,old_values)

    for _ in range(20):
  
        # model.train()

    
    
        
        
        npolicy,nvalues=model(status_set)
    
        
    
    
    
        
        critic_loss = (returns - nvalues).pow(2).sum()
    
    
        ratios = ((npolicy / old_policies) * action_set).sum(dim=1)
        clipped_ratios = torch.clamp(ratios, min=1.0-epsilon_clip, max=1.0+epsilon_clip).squeeze(0)
    
        actor_loss = -torch.min(ratios*advantages ,clipped_ratios*advantages ).sum()
    
        
        policy_entropy = (torch.log(npolicy) * npolicy).sum(1, keepdim=True).mean()
    
        loss = actor_loss + 0.5*critic_loss - 0.01* policy_entropy
        
        optimizer.zero_grad()

        
        loss.backward()
        

        optimizer.step()
   
    


         
