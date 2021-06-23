import gym
import random
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
writer=SummaryWriter()



class AC(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc= nn.Sequential(nn.Linear(4, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 128),)
        
        self.actor=nn.Linear(128,2)
        self.critic=nn.Linear(128,1)
               
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                
                
    def forward(self,x):
        out=self.fc(x)
       
       
        a=torch.nn.Softmax(dim=0)(self.actor(torch.relu(out)))
        c=self.critic(torch.relu(out))
     
        return a,c
   


device=torch.device('cuda:0')
     
model=AC().to(device)

# cmodel=AA().to(device)
env = gym.make('CartPole-v0')
env._max_episode_steps = 2000
status = env.reset()




def choose_action(status):
    global show

    model.eval()
    x=torch.tensor(status).to(torch.float32).to(device)
    A,C=model(x)
   
    a=torch.distributions.Categorical(A)
    
       

        
    action=int(a.sample().item())
    
    
    log_action=a.log_prob(torch.tensor(action*1.).to(device)).item()
    


    return action,log_action
 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
cliprange=0.1

for s in range(10000):
    exp=[]
    
    while True:
        done=False
        status = env.reset()  
        c=0
        while not done:
            c+=1
            experience=[]
            experience.append(status)
  
            action,log_action=choose_action(status)
    
            experience.append(action)
            status,reward,done,_=env.step(action)

            experience.append(done)

            experience.append(reward)
            experience.append(log_action)
            experience.append(status)

            exp.append(experience)
            
        print('count',c)
        writer.add_scalar('count',c,s)

        if len(exp)>128:
            break
            
   


    model.train()
    num_epochs=1
 

    for epoch in range(num_epochs):
        
        exp=random.sample(exp,128)
        for i in range(len(exp)):
            optimizer.zero_grad()

            status=torch.tensor(exp[i][0]).to(torch.float32).to(device)
            
            next_status=torch.tensor(exp[i][5]).to(torch.float32).to(device)
            
            done=torch.tensor(exp[i][2]).to(torch.float32).to(device)
            
            action=torch.tensor(exp[i][1]).to(torch.float32).to(device)
            
            log_action=torch.tensor(exp[i][4]).to(torch.float32).to(device)
            
            
            prob,s_value=model(status)
    
            _,n_s_value=model(next_status)
            
            target_value=1+(1-done)*0.8*n_s_value
    
    
            vloss=torch.nn.functional.smooth_l1_loss(s_value,target_value)
            
            
            
            # print(closs.item())
            
    
            a=torch.distributions.Categorical(prob)
    
    
    
            ratio = torch.exp(a.log_prob(action) -log_action)
            
    
  
            rarion_clamp = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) 
            
            ra = torch.min(ratio, rarion_clamp)
    
            aloss=-((target_value-s_value)*ra).sum()
            
            loss=aloss+vloss*10-a.entropy()*0.01
            
            
            loss.backward()
            optimizer.step()
        
    
    
    
    
    
         
