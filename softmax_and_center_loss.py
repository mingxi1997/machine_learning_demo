import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models


import random
import numpy as np
import torch.nn as nn
import torch
import os
import cv2
root='/home/xu/py_work/face3_t/'

names=[root+name for name in os.listdir(root)]

files=[]
for name in names:
    files.extend([name+'/'+n for n in os.listdir(name)])


class RES(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        # for param in self.model.parameters():
        #     param.requires_grad = False
        self.fc1=nn.Linear(1000,512)
        self.fc2=nn.Linear(512,len(names))
        self.center = Centerloss()
        
    def forward(self,x):
        out=self.model(x)
        
        
        out1=self.fc1(out)
        out2=self.fc2(out1)
 
        return out1,out2    


class Centerloss(nn.Module):
    def __init__(self, cls_num=len(names), feature_dim=512):
        
        super().__init__()
        self.center = nn.Parameter(torch.randn(cls_num, feature_dim), requires_grad=True)
        

    def forward(self, x, label, lambdas):
        
        
    

        center_exp = self.center[label]
        count = torch.histc(label, bins=len(names), max=len(names)-1, min=0)
        
        
        count_exp = count[label]
        
        
        a = torch.pow((x - center_exp), 2)  # N 10
        
        
        b = torch.sum(a, dim=1)  # N
        c = torch.div(b, count_exp.float())  # N
        d = torch.div(torch.sum(c),len(names))  # N
        loss = 1/2 * lambdas * d
        

        
        return loss
    
    
class my_dataset(Dataset):

    def __init__(self):
       self.files=files
       self.names=names
      
    def __len__(self):
       return len(self.files)

    def __getitem__(self, idx):
        
    
        pic=cv2.imread(self.files[idx]).transpose(2,0,1)
        
        direction=self.files[idx].split('/')[-2]
        
        names=[name.split('/')[-1] for name in  self.names]
        
        index=names.index(direction)
        
        
        return pic,index

device=torch.device('cuda:0')    
mydata=my_dataset()



model=RES().to(device)

# model.load_state_dict(torch.load('face_recs.pt'))    

learning_rate=0.01
batch_size=512
dataloader = DataLoader(mydata, batch_size=batch_size, shuffle=True, num_workers=4)


criterion = nn.CrossEntropyLoss()
optim_center = torch.optim.Adam(model.center.parameters(), lr=0.01)

class loss_set:
    def __init__(self):
        self.sum=0
        self.n=0
    def add(self,num):
        self.sum+=num
        self.n+=1
    def show(self):
        out=self.sum/self.n
        self.sum=0
        self.n=0
        return out
mloss=loss_set()
nloss=loss_set()

num_epochs=400
total_step=len(files)
optim_softmax = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    
  # learning_rate*=0.995  
 
  
  for i,(x,y) in enumerate(dataloader):
      # print(x.shape,y.shape,z.shape)
  
      optim_softmax.zero_grad()

      x=x.to(torch.float32).to(device)  
      out1,out2=model(x)
      
      y=y.to(device)
      loss_softmax=criterion(out2,y)
      
      loss_center = model.center(out1, y, 0.01)
      loss_center.backward(retain_graph=True)
      loss_softmax.backward()
      
      
      optim_center.step() 
      optim_softmax.step()
      
      
      # loss=loss_softmax.item()+loss_center.item()
      
      
      mloss.add(loss_softmax.item())
      nloss.add(loss_center.item())
      if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], softmax Loss: {:.4f},center Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, (i+1)*batch_size, total_step, mloss.show(),nloss.show()))
 
torch.save(model.state_dict(), 'face_recs.pt')
      










