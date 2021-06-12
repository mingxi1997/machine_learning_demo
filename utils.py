import numpy as np
import cv2
import torch.nn as nn
import torch
import random

import numpy as np

def gen_values(num):
    x=[]
    for i in range(num):
        x.append(pow(0.99,i))
    
    r=np.zeros(num)
    for i in range(num):
      for j in range(num-i):
        r[i]+=x[num-j-1]
    return r

def get_screen(env):
    screen = env.render(mode='rgb_array')
    screen= screen[150:350,100:500,:]
    screen = cv2.cvtColor(screen,cv2.COLOR_RGB2GRAY)
    screen=cv2.resize(screen, (80,40))
    return screen



def pad(mlist,num):
    pad_num=num-len(mlist)
    new_list=[]
    for i in range(pad_num):
        new_list.append(mlist[0])
    new_list.extend(mlist)
    return new_list
    


class ShortMem:
    def __init__(self,lenth):
        self.mem=[]
        self.lenth=lenth
        
    def __len__(self):
        return len(self.mem)
    def get_mem(self,screen):
        self.mem.append(screen)
        if len(self.mem)>self.lenth:
            self.mem=self.mem[1:]
    def show(self):
        if len(self.mem)==4:
            pass
            
        else:
            self.mem=pad(self.mem,4)
        return np.stack(self.mem)
    
class Experiences():
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
    def __len__(self):
        return len(self.mem)
    def sample(self,batch):
        sample = random.sample(self.mem, batch)
        return sample
            
            
def collect_experience(env,model,experiences,short_mem):
    done=False

    env.reset()
    c=0
    s_set=[]
    a_set=[]
    while not done:
        c+=1
        short_mem.get_mem(get_screen(env))
        
        s=torch.tensor(short_mem.show()).unsqueeze(0)
        s_set.append(s)
        
        y=model(s)[0]
        
        y=nn.Softmax(dim=-1)(y)
        
        a=torch.distributions.Categorical(y)
        
        action=int(a.sample().item())
        a_set.append(action)
        
        status,reward,done,_=env.step(action)
        
        # short_mem.get_mem(get_screen(env))
        
        # ns=torch.tensor(short_mem.show()).unsqueeze(0)
    rewards_set= gen_values(len(s_set))    
    for i in range(len(s_set)):
        experiences.push([s_set[i],a_set[i],rewards_set[i]])
        
      
        
    return c,experiences            
            
            
def s_collect_experience(env,model,experiences,short_mem):
    done=False

    status=env.reset()
    c=0
    s_set=[]
    a_set=[]
    while not done:
        c+=1
        
        
        s=torch.tensor(status)
        
        s_set.append(s)
        y=model(s)[0]
        
        y=nn.Softmax(dim=-1)(y)
        
        a=torch.distributions.Categorical(y)
        
        action=int(a.sample().item())
        
        a_set.append(action)
        
        status,reward,done,_=env.step(action)
        
        short_mem.get_mem(get_screen(env))
        
        # ns=torch.tensor(status)
        
        # ns_set.append(ns)
        
    rewards_set=    gen_values(len(s_set))
    for i in range(len(s_set)):
        experiences.push([s_set[i],a_set[i],rewards_set[i]])
        
        
        
    print('count :{}'.format(c))
        
    return c,experiences                        
            
            
            
            
            
            
            
            
            
            
            
            
            
            