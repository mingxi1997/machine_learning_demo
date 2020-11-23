import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import torch.nn as nn
import torch
x=pd.read_csv('../data/mnist/train.csv')

image_set=[]

for i in range(10):
    image_set.append(x[x.label==i+1].drop(['label'],axis=1).values)
    
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(1,32,3,2,1)
        self.n1=nn.BatchNorm2d(32)
        
        self.c2=nn.Conv2d(32,64,3,2,1)
        self.n2=nn.BatchNorm2d(64)
        
        self.c3=nn.Conv2d(64,64,3,3,1)
        self.n3=nn.BatchNorm2d(64)
    
        self.fc1=nn.Linear(576,128)

    def forward(self,x):
        out=self.c1(x)
        
        out=self.n1(out)
        out=nn.functional.relu(out)
        
        out=self.c2(out)
        out=self.n2(out)
        out=nn.functional.relu(out)
        
        out=self.c3(out)
        out=self.n3(out)
        out=nn.functional.relu(out)

        out=out.view(out.shape[0],-1)
        out=self.fc1(out)
 
        return out    
class my_dataset(Dataset):

    def __init__(self):
       self.data = image_set
      
    def __len__(self):
       return 25600

    def __getitem__(self, idx):
        # img_name = os.listdir('last')[idx]
    
        index=random.sample(range(0,9),2)
        my_index=index[0]
        his_index=index[1]
        
        my_pic_index=random.sample(range(0,len(image_set[my_index])),2)
        his_pic_index=random.sample(range(0,len(image_set[his_index])),1)
        
        my_pic1=image_set[my_index][my_pic_index[0]].reshape(1,28,28)
        my_pic2=image_set[my_index][my_pic_index[1]].reshape(1,28,28)

        his_pic=image_set[his_index][his_pic_index[0]].reshape(1,28,28)
        
        
        return my_pic1,my_pic2,his_pic
  
    
device=torch.device('cuda:0')
mydata=my_dataset()
model=CNN().to(device)
learning_rate=0.01

dataloader = DataLoader(mydata, batch_size=128, shuffle=True, num_workers=4)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

for i in range(100):
  for x,y,z in dataloader:
      # print(x.shape,y.shape,z.shape)
  
      optimizer.zero_grad()

      x=x.to(torch.float32).to(device)    
      y=y.to(torch.float32).to(device)      
      z=z.to(torch.float32).to(device)      
    

      out1=model(x)
      out2=model(y)
      out3=model(z)
      
     
      
      loss=triplet_loss(out1,out2,out3)
     
      if type(loss)==int:
         pass
      else:
        # loss=loss.sum()
        print(loss.item())
        loss.backward()
        optimizer.step()




testloader = DataLoader(mydata, batch_size=1, shuffle=True, num_workers=4)

model.eval()

for x,y,z in testloader:
        x=x.to(torch.float32).to(device)    
        y=y.to(torch.float32).to(device)      
        z=z.to(torch.float32).to(device)      
    

        out1=model(x)
        out2=model(y)
        out3=model(z)
        print(torch.dist(out1,out2).item()<torch.dist(out1,out3).item())
     
    
    










