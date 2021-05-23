import gym
import random
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
import random
import torch.nn.functional as F

# torch.manual_seed(1423)

class AC(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1= nn.Linear(3, 128)
        
        self.fc2= nn.Linear(128, 128)
        
        self.fc3= nn.Linear(128, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    def forward(self,x):
        
        out=self.fc1(x)
        out=torch.relu(out)
        
        out=self.fc2(out)
        out=torch.relu(out)
        
        out=self.fc3(out)
        out=torch.tanh(out)*2
       
        return out

class CR(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1= nn.Linear(4, 128)
 
        self.fc2= nn.Linear(128, 128)
        
        self.fc3=nn.Linear(128,1)
               
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                
                
    def forward(self,x):
       
        out=self.fc1(x)
        out=torch.relu(out)
        
        out=self.fc2(out)
        out=torch.tanh(out)

        out=self.fc3(out)
      

   
     
        return out
   
class loss_set:
    def __init__(self):
        self.sum=0
        self.n=0
    def add(self,num):
        self.sum+=num
        self.n+=1
    def show(self):
        if self.n==0:
            return 0
        else:
          out=self.sum/self.n
          self.sum=0
          self.n=0
          return out

def choose_action(status):
    actor.eval()
    x=torch.tensor(status).to(torch.float32).to(device)
    y=actor(x)
    
    

    return y  


device=torch.device('cuda:0')

actor= AC().to(device)
target_actor=AC().to(device)

critic=CR().to(device)
target_critic=CR().to(device)


target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())


target_actor.eval()
target_critic.eval()

env = gym.make('Pendulum-v0')
env._max_episode_steps = 200



learning_rate=0.001
mloss=loss_set()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
coptimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate)


class memory_store():
    def __init__(self):
        self.mem=[]
        self.count=0
    def push(self,memory):
        if self.count<10000:
            self.mem.append(memory)
            self.count+=1
        else:
            self.mem=self.mem[1:]
            self.mem.append(memory)
          
            
memory_set=memory_store()
        
mloss=loss_set()

e=0.5

tau=0.02

def soft_update(net,net_target,  tau):
            for target_param, param  in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

experiences=[]
for episode in tqdm(range(100)):
    
    
      status=env.reset()
      count=0
      
      for i in range(200):
        experience=[]    
   
        experience.append(status)  
        action=choose_action(status)
    
        experience.append(action)
        status,reward,done,_=env.step(action.cpu().detach().numpy())

        experience.append(reward)
        experience.append(status)
        memory_set.push(experience)
        count+=reward
        if len(memory_set.mem)>32:
   
        
            sample = random.sample(memory_set.mem, 32)
            
            s = torch.tensor([exp[0] for exp in sample]).float().to(device)
            a = torch.tensor([exp[1] for exp in sample]).float().to(device).unsqueeze(1)
            r = torch.tensor([exp[2] for exp in sample]).float().to(device)
            ns = torch.tensor([exp[3] for exp in sample]).float().to(device)
 

  
            critic.train()

            target_action=target_actor(ns).detach()
            next_critic=target_critic(torch.cat((ns,target_action),dim=1)).squeeze()
            target_q=r+0.99*next_critic.detach()
        
     
            predict_q=critic(torch.cat((s,a),dim=1)).squeeze()
            
            critic_loss=torch.nn.MSELoss()(predict_q,target_q)
            # print('critic: ',critic_loss.item())
        
        # print('critic: ',critic_loss.item())
        
            coptimizer.zero_grad()

            critic_loss.backward()
            coptimizer.step()
        
        
          
       

            actor.train()
        
        
            predict_action=actor(s)
        
            q=critic(torch.cat((s,predict_action),dim=1)).squeeze()
        
        
            actor_loss=torch.mean(-q)
        
            #print('actor: ',actor_loss.item())
            optimizer.zero_grad()
            actor_loss.backward()
            optimizer.step()    
    
            soft_update(actor, target_actor, tau)
            soft_update(critic, target_critic, tau)
      print(count)
                
                
