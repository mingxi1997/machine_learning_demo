import gym
import random
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
import random
import torch.nn.functional as F

torch.manual_seed(1423)

class NN(nn.Module):  
    def __init__(self):
        super().__init__()
        self.fc1= nn.Linear(4, 64)
        self.fc3= nn.Linear(64, 2)
    def forward(self,x):
        out=self.fc1(x)
        out=torch.tanh(out)
        out=self.fc3(out)
        # out=torch.sigmoid(out)
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

def e_greedy_nn(status,e):
    predict_model.eval()
    x=torch.tensor(status).to(torch.float32).to(device)
    y=predict_model(x)
    
    action=np.argmax(y.detach().cpu().numpy()) if torch.rand(1, ).item() > e else torch.randint(0, 2,(1,)).item()

    return action   


def last(status):
    predict_model.eval()
    x=torch.tensor(status).to(torch.float32).to(device)
    y=predict_model(x)
    
    action=np.argmax(y.detach().cpu().numpy())

    return action   



device=torch.device('cuda:0')

predict_model = NN().to(device)
target_model=NN().to(device)
target_model.load_state_dict(predict_model.state_dict())
target_model.eval()
env = gym.make('CartPole-v0')
env._max_episode_steps = 200



learning_rate=0.001
mloss=loss_set()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(predict_model.parameters(), lr=learning_rate)


class memory_store():
    def __init__(self):
        self.mem=[]
        self.count=0
    def push(self,memory):
        if self.count<256:
            self.mem.append(memory)
            self.count+=1
        else:
            self.mem=self.mem[1:]
            self.mem.append(memory)
          
            
memory_set=memory_store()
        
mloss=loss_set()

e=1



def collect_experience(nums):
  c=0
  done=False
  while c<nums: 
    experience=[]
    if done or c==0:
        status=env.reset()
        experience.append(status)
    else:
        experience.append(status)
    action=e_greedy_nn(status,e)
    experience.append(action)
    status,reward,done,_=env.step(action)
    experience.append(reward)
    experience.append(status)
    memory_set.push(experience)
    c+=1

collect_experience(128)

for iteration in tqdm(range(10000)):
    collect_experience(128)
    iteration+=1
    if iteration%5==0:
        target_model.load_state_dict(predict_model.state_dict())

    if e>0.05:
        e-=1/5000
    
    for i in range(4):
        sample = random.sample(memory_set.mem, 16)
        s = torch.tensor([exp[0] for exp in sample]).float().to(device)
        a = torch.tensor([exp[1] for exp in sample]).float().to(device)
        rn = torch.tensor([exp[2] for exp in sample]).float().to(device)
        sn = torch.tensor([exp[3] for exp in sample]).float().to(device)

        predict_model.train()

        qp=predict_model(s)
        
        #print(qp)
        
        predict,_=torch.max(qp, axis=1)
        
        
        next_q=torch.max(target_model(sn),axis=1).values*0.95+rn
   #     with torch.no_grad():
   #         qpp=target_model(sn)
    #    next_q, _ = torch.max(qpp, axis=1)
  #      target = next_q*0.95+rn
        target=next_q.detach()
        
      #  print(predict)
   #     print(target)

        loss=criterion(predict,target)
        optimizer.zero_grad()
        #print(loss.item())
        loss.backward()
        optimizer.step()
     
  
predict_model.eval()
status=env.reset()
done=False
count=0
    
while not done:
        count+=1
        env.render()
        action=last(status)
       
        status, reward, done, info = env.step(action.item())
 
print(count)
torch.save(predict_model.state_dict(), 'our_model_3.pt')
