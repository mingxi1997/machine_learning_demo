import gym
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import numpy as np
import cv2

action_repeat=4
img_stack=4


class Env():
   

    def __init__(self):
        self.env = gym.make('PongNoFrameskip-v4')
        self.stack=[]


    def reset(self):
        
        status=self.env.reset()
        status=self.crop_and_resize(status)


        
        
      
        
        
        self.stack = [status] * img_stack  # four frames for decision
        return np.stack(self.stack)

    def step(self, action):
        total_reward = 0
      
        for i in range(action_repeat):
            status, reward, done, _ = self.env.step(action)
            status=self.crop_and_resize(status)
            
            
            total_reward += reward

     
            self.stack.pop(0)
            self.stack.append(status)
            # don't penalize "die state"
            if done:
                total_reward =-1
                break


        return np.stack(self.stack), total_reward, done,len((self.stack))
    @staticmethod
    def crop_and_resize(status):
        status=status[:25,:,:]
        status=cv2.resize(status, (84,84))
        status = cv2.cvtColor(status,cv2.COLOR_RGB2GRAY)
        return status
        
        
    
    # def get_screen(self):
    #     screen = self.env.render(mode='rgb_array')
    #     screen= screen[150:300,225:375,:]
    #     screen = cv2.cvtColor(screen,cv2.COLOR_RGB2GRAY)
    #     screen=cv2.resize(screen, (75,75))
        
    #     return screen

    def render(self):
        self.env.render()

class AC(nn.Module):  
    def __init__(self,input_shape,n_action):
        super().__init__()
        self.input_shape=input_shape
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
         )
        conv_out_size = self._get_conv_out(input_shape)

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
        out = x.float() / 127-1
        
        out=self.conv(out)
        out=out.view(x.shape[0],-1)
        return torch.nn.Softmax(dim=-1)(self.actor(out)),self.critic(out)
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

device=torch.device('cuda:0')
action_space_num=6
model=AC((img_stack, 84, 84),action_space_num).to(device)
env = Env()
env._max_episode_steps = 2000
status = env.reset()
gamma = 0.99
lambda_gae = 0.96

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

running_score=-21
epsilon_clip=0.1
ciritic_coefficient = 0.5
entropy_coefficient = 0.01
ppo_epoch=30
writer=SummaryWriter()
max_grad_norm=0.5


num_workers=5

def get_advantage(rewards,values):


    assert old_values.dim()==1
    assert rewards.dim()==1
        
        
    running_tderror=torch.zeros_like(rewards)
    advantages= torch.zeros_like(rewards)

       
    for t in reversed(range(len(rewards))):
         if t==len(rewards)-1:
            running_tderror[t]=rewards[t]-values[t]
            advantages[t]=running_tderror[t]
           
         else:
             running_tderror[t]=rewards[t]+gamma*values[t+1]-values[t]
             advantages[t]=running_tderror[t]+(gamma * lambda_gae)*advantages[t+1]
             
    returns=advantages+values
    returns=(returns-returns.mean())/returns.std()
    advantages=(advantages-advantages.mean())/advantages.std()
    return returns,advantages



def choose_action(status):

    with torch.no_grad():
        x=torch.tensor(status).to(torch.float32).to(device).unsqueeze(0)
        policy,value=model(x)
        dist=torch.distributions.Categorical(policy)
        action=int(dist.sample().item())
        return action,value,policy
 



for s in range(100000):
        
   
    returns=torch.tensor([]).to(device)
    advantages=torch.tensor([]).to(device)
    status_set=torch.tensor([]).to(device)
    action_set=torch.tensor([]).to(device)
    old_policies=torch.tensor([]).to(device)
    old_values=torch.tensor([]).to(device)
    advantages=torch.tensor([]).to(device)
    c=0
    for _ in range(num_workers):
    
        exp=[]
        done=False
        status = env.reset()  
        
        while not done:
           
            
            
            experience=[]
            experience.append(status)
      
            action,value,policy=choose_action(status)
            
            action_one_hot = torch.zeros(action_space_num)
            action_one_hot[action] = 1
    
            experience.append(action_one_hot)
          
            status,reward,done,_=env.step(action)

                
            experience.append(reward)
            experience.append(policy)
            experience.append(value)
            exp.append(experience)
            
            
            c+=reward
        
                
        
       
        
      
                
       
        
        nexp=list(zip(*exp))
        
       
        status_set_=torch.tensor(nexp[0]).to(torch.float32).to(device)
        action_set_=torch.stack(nexp[1]).to(torch.float32).to(device)
        reward_set_=torch.tensor(nexp[2]).to(torch.float32).to(device)
        old_policies_=torch.stack(nexp[3]).to(torch.float32).to(device)
        old_values_=torch.stack(nexp[4]).to(torch.float32).to(device).squeeze()
        
       
        returns_,advantages_ =get_advantage(reward_set_, old_values_)
    
        returns=torch.cat((returns,returns_),dim=0)
        advantages=torch.cat((advantages,advantages_),dim=0)
        status_set=torch.cat((status_set,status_set_),dim=0)
        action_set=torch.cat((action_set,action_set_),dim=0)
        old_policies=torch.cat((old_policies,old_policies_),dim=0)
        old_values=torch.cat((old_values,old_values_),dim=0)
        
    score=c/num_workers
    running_score = 0.99 * running_score + 0.01 * score
    print('epoch :{} score :{} running_score: {}'.format(s,score,running_score))
    
    writer.add_scalar('c',c,s)
    writer.add_scalar('running_score',running_score,s)
   
       
        
  

    for _ in range(ppo_epoch):
  
       

        npolicy,nvalues=model(status_set)
    
        critic_loss = (returns - nvalues.squeeze()).pow(2).mean()
    
    
        ratios = ((npolicy / old_policies.squeeze()) * action_set).sum(dim=1)
        clipped_ratios = torch.clamp(ratios, min=1.0-epsilon_clip, max=1.0+epsilon_clip).squeeze(0)
        actor_loss = -torch.min(ratios*advantages ,clipped_ratios*advantages ).mean()
        policy_entropy = (torch.log(npolicy) * npolicy).sum(1, keepdim=True).mean()
        loss = actor_loss + 0.5*critic_loss - 0.01* policy_entropy
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
   
    


    
         
