import gym
import random
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
import random
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
from PIL import Image


device=torch.device('cuda:0')

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])
env = gym.make('Pendulum-v0')
env._max_episode_steps = 200


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # encoder part
        
        self.cnn_en=nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 16, 6, 3, 3), 
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(16, 16, 6, 3, 3), 
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 4, 2, 2),   # B,  32, 28, 28
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 4, 2, 2),  # B,  32, 14, 14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  7, 7
            nn.LeakyReLU(0.2, inplace=True),

        )
        # self.fc1 = nn.Linear(64*7*7, 512)
        # self.fc2 = nn.Linear(512, 256)
        
        self.fc31 = nn.Linear(64*6*6, 3)
        self.fc32 = nn.Linear(64*6*6, 3)
        # decoder part
        self.fc4 = nn.Linear(3, 64*6*6)
        # self.fc5 = nn.Linear(256, 512)
        # self.fc6 = nn.Linear(512, 784)
        
        self.cnn_de = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  64,  14,  14
            nn.LeakyReLU(0.2, inplace=True),
            # nn.ConvTranspose2d(32, 32, 4, 2, 1,1), # B,  32, 28, 28
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 32, 4, 2, 2),   # B, 1, 28, 28
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 2,1), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(16, 3, 6, 3, 3,2), 
            nn.LeakyReLU(0.2, inplace=True),
            
        )
        
        
        
    def encoder(self, x):
     
        h=  self.cnn_en(x)
 
        h= h.view(h.size(0),-1)
        
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
      
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = self.fc4(z).view(-1, 64, 6, 6)
         
        h=self.cnn_de(h)
  
        
        return torch.sigmoid(h) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        
        z = self.sampling(mu, log_var)

        return self.decoder(z), mu, log_var
    
vae= VAE().to(device)
vae.load_state_dict(torch.load('ddpg.pt'))


def get_screen():
    
    screen = env.render(mode='rgb_array')
    screen = screen[175:375,150:350]
    pic=cv2.resize(screen,(128,128))
    # pic=pic.transpose(2,0,1)
    pic=transform(pic)
    
    
    return pic
   

class AC(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1= nn.Linear(6, 128)
        
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
        self.fc1= nn.Linear(7, 128)
 
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
    
    status=status.float().to(device).unsqueeze(0)
    # print(status.shape)
    mu, log_var = vae.encoder(status)
        
    s = vae.sampling(mu, log_var)
   
    y=actor(s)
    
    

    return y  



actor= AC().to(device)
target_actor=AC().to(device)

critic=CR().to(device)
target_critic=CR().to(device)


target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())


target_actor.eval()
target_critic.eval()





learning_rate=0.001
mloss=loss_set()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
coptimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate)
aoptimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)


class memory_store():
    def __init__(self):
        self.mem=[]
        self.count=0
    def push(self,memory):
        if self.count<2000:
            self.mem.append(memory)
            self.count+=1
        else:
            self.mem=self.mem[1:]
            self.mem.append(memory)
          
            
memory_set=memory_store()
        
mloss=loss_set()

e=0.5




def pics_to_vec(pics):
    s_set=[]
    for p in pics:
                s1,s2=p
                mu, log_var = vae.encoder(s1.unsqueeze(0).to(device))
        
                s1_ = vae.sampling(mu, log_var).squeeze()
            
                mu, log_var = vae.encoder(s2.unsqueeze(0).to(device))
        
                s2_ = vae.sampling(mu, log_var).squeeze()
                
                s=torch.cat((s1_,s2_))
                
                s_set.append(s)
                
    status=torch.stack(s_set)
    return status


tau=0.02

def soft_update(net,net_target,  tau):
            for target_param, param  in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

experiences=[]
for episode in tqdm(range(100)):
    
     

   
    
      status=env.reset()
      count=0
      sc_set=[]
      for i in range(200):
          
        sc_set.append(get_screen())  
        experience=[]    
        if len(sc_set)!=1:
            experience.append((sc_set[-1],sc_set[-2]))
        else:
            experience.append((sc_set[-1],sc_set[-1]))
        
        action=choose_action(get_screen())
    
        experience.append(action)
        status,reward,done,_=env.step(action.cpu().detach().numpy())

        experience.append(reward)
        experience.append((get_screen(),sc_set[-1]))
        count+=reward
        
        memory_set.push(experience)

        if len(memory_set.mem)>200 :
            vae.train()

            aoptimizer.zero_grad()
            optimizer.zero_grad()
            coptimizer.zero_grad()

 
        
            sample = random.sample(memory_set.mem, 32)
            
            
            pics=[exp[0] for exp in sample]
            status=pics_to_vec(pics)
           
            

            
            
            
            a = torch.tensor([exp[1] for exp in sample]).float().to(device).unsqueeze(1)
            r = torch.tensor([exp[2] for exp in sample]).float().to(device)
            npics = [exp[3] for exp in sample]
            nstatus=pics_to_vec(npics)
 
 
  
            critic.train()

            target_action=target_actor(nstatus).detach()
            next_critic=target_critic(torch.cat((nstatus,target_action),dim=1)).squeeze()
            target_q=r.squeeze()+0.99*next_critic.detach()
        
     
            predict_q=critic(torch.cat((status,a),dim=1)).squeeze()
            
            critic_loss=torch.nn.MSELoss()(predict_q,target_q)
            # print('critic: ',critic_loss.item())
        
        # print('critic: ',critic_loss.item())
        
            coptimizer.zero_grad()

            critic_loss.backward()
            coptimizer.step()
        
        
          
       

            actor.train()
        
        
            predict_action=actor(status)
        
            q=critic(torch.cat((status,predict_action),dim=1)).squeeze()
        
        
            actor_loss=torch.mean(-q)
        
            #print('actor: ',actor_loss.item())
            optimizer.zero_grad()
            actor_loss.backward()
            optimizer.step()    
    
            soft_update(actor, target_actor, tau)
            soft_update(critic, target_critic, tau)
      print(count)                


                
