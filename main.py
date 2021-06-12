import torch
import torch.nn as nn
import gym
import cv2
from network import AC
from utils import ShortMem,Experiences,get_screen,collect_experience
import  matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
writer=SummaryWriter()

ENTROPY_BETA =10

LEARNING_RATE=1e-6
experiences=Experiences()   
env = gym.make('CartPole-v0')
env._max_episode_steps = 200
count=0
discount=0.9
device=torch.device('cuda:0')
model=AC((4, 40, 80),2).to(device)
CLIP_GRAD=0.5    
batch_size=128
short_mem=ShortMem(4)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-3)
model.eval()

while len(experiences)<256:
        c,experiences=collect_experience(env,model.cpu(),experiences,short_mem)
print('start_train')
for e in range(1000000):
    model.eval()
    c,experiences=collect_experience(env,model.cpu(),experiences,short_mem)
    
    model.to(device).train()
    optimizer.zero_grad()
    
    

    samples= experiences.sample(batch_size)
    s = torch.cat([exp[0] for exp in samples]).to(device)
    a = torch.tensor([exp[1] for exp in samples]).to(device)
    value_rev = torch.tensor([exp[2] for exp in samples]).to(torch.float32).to(device)
  
    
    
    logit,value=model(s)
    
    

    loss_value_v = torch.nn.functional.mse_loss(value.squeeze(), value_rev)/16
    
    adv_v = (value_rev - value.squeeze())
    
    
    log_prob_v = torch.log_softmax(logit, dim=1)
    
    log_prob_v_actions=log_prob_v[range(batch_size),a]
    
    loss_policy_v=-(adv_v*log_prob_v_actions).mean()
    
    loss_policy_v.backward(retain_graph=True)
    
    prob_v = torch.softmax(logit, dim=1)
    
    entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()
    
    
    
    loss_v = entropy_loss_v + loss_value_v

    writer.add_scalar('count',c,e)
    writer.add_scalar('entropy_loss_v',entropy_loss_v.item(),e)
    writer.add_scalar('loss_value_v',loss_value_v.item(),e)
    writer.add_scalar('loss_v',loss_v.item(),e)
    writer.add_scalar('loss_policy_v',loss_policy_v.item(),e)
    loss_v.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
    optimizer.step()





    
    
    
    
    
    
    