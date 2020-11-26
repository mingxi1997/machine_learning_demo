import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import torch.nn as nn
import torch
import os
from PIL import Image
import torchvision.models as models

from tqdm import tqdm

root='/home/xu/py_work/face2/'
names=os.listdir(root)

class RES(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        # for param in self.model.parameters():
        #     param.requires_grad = False
        self.fc=nn.Linear(1000,128)
        self.net=Center()
        
    def forward(self,x):
        out=self.model(x)
        out=self.fc(out)
        return out    


    
class my_dataset(Dataset):

    def __init__(self):
       self.root=root
       self.names=names
       
      
    def __len__(self):
       return 111714

    def __getitem__(self, idx):
        # img_name = os.listdir('last')[idx]
    
        index=random.sample(range(0,len(self.names)),2)
        
        my_index=index[0]
        his_index=index[1]
        
        my_root=self.root+self.names[my_index]
        his_root=self.root+self.names[his_index]
        
        my_pic_index=random.sample(range(0,len(os.listdir(my_root))),2)
        his_pic_index=random.sample(range(0,len(os.listdir(his_root))),1)
        
      
        
        my_pic1=self.root+self.names[my_index]+'/'+os.listdir(my_root)[my_pic_index[0]]
        my_pic2=self.root+self.names[my_index]+'/'+os.listdir(my_root)[my_pic_index[1]]
        his_pic=self.root+self.names[his_index]+'/'+os.listdir(his_root)[his_pic_index[0]]
        
        
        m1=np.array(Image.open(my_pic1).convert('RGB')).reshape(3,120,120)
        m2=np.array(Image.open(my_pic2).convert('RGB')).reshape(3,120,120)
        h=np.array(Image.open(his_pic).convert('RGB')).reshape(3,120,120)
        
        return m1,m2,h,my_index
  


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
    



class Center(nn.Module):
    def __init__(self, cls_num=len(names), feature_dim=128):
        
        super().__init__()
        self.center = nn.Parameter(torch.randn(cls_num, feature_dim), requires_grad=True)
        self.cls_num=cls_num

    def forward(self, x, label, lambdas):
        
        global show
        show=x,label,lambdas
        
      
      
        center_exp = self.center[label]
        
        count = torch.histc(label, bins=self.cls_num+1, max=self.cls_num, min=0)
        
        
        count_exp = count[label]
        
        
        a = torch.pow((x - center_exp), 2)  # N 10
        
        b = torch.sum(a, dim=1)  # N
        c = torch.div(b, count_exp.float())  # N
        d = torch.div(torch.sum(c),self. cls_num)  # N
        loss = 1/2 * lambdas * d
        

        
        return loss
    
global show    
mloss=loss_set()
num_epochs=100
total_step=111714    
my_data=my_dataset()

device=torch.device('cuda:0')
mydata=my_dataset()
model=RES().to(device)

# center=Center(len(names),128)


# model.load_state_dict(torch.load('face_rec90.pt'))

learning_rate=0.01

dataloader = DataLoader(mydata, batch_size=256, shuffle=True, num_workers=4)

criterion = nn.CrossEntropyLoss()

triple_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
center_optimizer=torch.optim.SGD(model.net.parameters(), lr=0.5, momentum=0.9)


triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)



for epoch in range(num_epochs):
  for i,(x,y,z,index) in enumerate(dataloader):
      # print(x.shape,y.shape,z.shape)
      
      triple_optimizer.zero_grad()

      x=x.to(torch.float32).to(device)    
      y=y.to(torch.float32).to(device)      
      z=z.to(torch.float32).to(device)      
      index=index.to(device)  

      out1=model(x)
      out2=model(y)
      out3=model(z)
      
     
      loss_center = model.net(out1, index,0.5)
      
      loss=triplet_loss(out1,out2,out3)
      
      mloss.add(loss.item()+loss_center.item())
      
      if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, (i+1)*512, total_step, mloss.show()))
      # if type(loss)==int:
      #     pass
      # else:
        # loss=loss.sum()
      loss_center.backward(retain_graph=True)
      loss.backward()
      
      mloss.add(loss_center+loss)
      center_optimizer.step()
      triple_optimizer.step()
      
      
  if epoch%10==0:
      torch.save(model.state_dict(), 'face_rec{}.pt'.format(epoch))


testloader = DataLoader(mydata, batch_size=1, shuffle=True, num_workers=4)



model.eval()
times=0
count=0
for x,y,z in tqdm(testloader):
        times+=1
        x=x.to(torch.float32).to(device)    
        y=y.to(torch.float32).to(device)      
        z=z.to(torch.float32).to(device)      
    

        out1=model(x)
        out2=model(y)
        out3=model(z)
        if (torch.dist(out1,out2).item()<torch.dist(out1,out3).item())==True:
            count+=1
        if times==10000:
            print('count :{},times :{}'.format(count,times))
        # print(torch.dist(out1,out2).item()<torch.dist(out1,out3).item())
print('ratio:  ',count/times)
     





