import gym
import random
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
import cv2

action_repeat=1
img_stack=4



class Env():
    """
    Environment wrapper for CarRacing 
    """

    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.env._max_episode_steps = 2000
        self.stack=[]


    def reset(self):
        
        self.env.reset()
        screen=self.get_screen()
        self.stack = [screen] * img_stack  # four frames for decision
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
      
        for i in range(action_repeat):
            status, reward, done, _ = self.env.step(action)
            
            
            total_reward += reward

     
            self.stack.pop(0)
            self.stack.append(self.get_screen())
            # don't penalize "die state"
            if done:
                total_reward =-1
                break


        return np.array(self.stack), total_reward, done,len((self.stack))
    
    def get_screen(self):
        screen = self.env.render(mode='rgb_array')
        screen= screen[150:350,100:500,:]
        screen = cv2.cvtColor(screen,cv2.COLOR_RGB2GRAY)
        screen=cv2.resize(screen, (80,40))
        return screen

    def render(self):
        self.env.render()


class AC(nn.Module):  
    def __init__(self,input_shape,n_action):
        super().__init__()
        
        self.conv=nn.Sequential(
            nn.Conv2d(input_shape[0],32,8,4),
            # nn.BatchNorm2d(32),
            nn.ELU(),
            
            nn.Conv2d(32,64,4,2),
            # nn.BatchNorm2d(64),
            nn.ELU(),
            
            nn.Conv2d(64,64,3,1),
            # nn.BatchNorm2d(64),
            nn.ELU()
            
            )
        conv_out_size = self._get_conv_out(input_shape)
        print('model initialized ,conv_out_size :{} ,output action :{}'.format(conv_out_size,n_action))

        self.actor=nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ELU(),
            nn.Linear(128, n_action)            
            )
        self.critic=nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ELU(),
            nn.Linear(128, 1)            
            )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
     
    def forward(self,x):
        out = x.float() / 256
        out=self.conv(out)
        out=out.view(out.size(0),-1)      
        return torch.nn.Softmax(dim=-1)(self.actor(out)),self.critic(out)
    
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1,*shape))
        return int(np.prod(o.size()))


    

env=   Env()




    
def sample(status_set, action_set, returns, advantages, old_policies,old_values):
    
    
    index=random.sample(list(range(len(status_set))),int(len(status_set)*0.8))
    return status_set[index], action_set[index], returns[index], advantages[index], old_policies[index],old_values[index]


device=torch.device('cuda:0')
      
model=AC((img_stack, 40, 80),2).to(device)

model.load_state_dict(torch.load("save.pt"))
gamma = 0.99
lambda_gae = 0.96
lr=1e-7
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
running_score=0

epsilon_clip=0.1
ciritic_coefficient = 0.5
entropy_coefficient = 0.01
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

    model.eval()
    x=torch.tensor(status).to(torch.float32).to(device).unsqueeze(0)
    A,C=model(x)
    
    a=torch.distributions.Categorical(A)
    
        

        
    action=int(a.sample().item())
    

    return action,A
  


record=0
for s in range(10000):
  try:
    # epsilon_clip*=0.99
    exp=[]
    
    
    done=False
    status = env.reset()  
    c=0
    while not done:
        c+=1
        
        
        experience=[]
        experience.append(status)
  
    
  
        model.eval()
        
        # x=torch.tensor(status).to(torch.float32).to(device).unsqueeze(0)
        # actor,_=model(x)
        
        # a=torch.distributions.Categorical(actor)
    
        

        
        # action=int(a.sample().item())
  
    
  
    
  
       
  
        action,_=choose_action(status)
        
        
        
        
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
    if c>record:
        print('save',c)
        record=c
        torch.save(model.state_dict(),'save.pt')
        
    score=c
    running_score = 0.99 * running_score + 0.01 * score
    # print(running_score)
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

    # print(epsilon_clip)

    for t in range(40):
        
  
        # model.train()


        npolicy,nvalues=model(status_set)
    
        
        critic_loss = (returns - nvalues.squeeze()).pow(2).sum()
    
    
        ratios = ((npolicy / old_policies) * action_set).sum(dim=1)
        clipped_ratios = torch.clamp(ratios, min=1.0-epsilon_clip, max=1.0+epsilon_clip).squeeze(0)
        actor_loss = -torch.min(ratios*advantages ,clipped_ratios*advantages ).sum()
    
        
        policy_entropy = (torch.log(npolicy) * npolicy).sum(1, keepdim=True).mean()
        
  
    
        loss = actor_loss + 0.5*critic_loss - 0.01* policy_entropy
    
        
        optimizer.zero_grad()

        
        loss.backward()
        

        optimizer.step()
    writer.add_scalar('actor_loss',actor_loss,s)
    writer.add_scalar('critic_loss',critic_loss,s)
    writer.add_scalar('policy_entropy',policy_entropy,s)
  except:
      pass



    


