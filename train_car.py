import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from torch.utils import data
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import random


# root='~/Downloads/dataset/IMG/'

root='/home/xu/save/'

docs=[root+d+'/'+'driving_log.csv' for d in os.listdir(root)]
# folders=[root+d+'/'+'IMG/' for d in os.listdir(root)]

record=[]
for doc in docs:
    record.append(pd.read_csv(doc,names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']))



old_data_index=pd.concat(record,axis=0,ignore_index=True)


d2=pd.read_csv('~/Downloads/track2data/'+'driving_log.csv',names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
# d3=pd.read_csv('~/Downloads/dataset/'+'driving_log.csv',names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

old_data_index=pd.concat([d2,old_data_index],axis=0,ignore_index=True)

nozero_set=[]
zero_set=[]
for i in range(len(old_data_index)):
    if old_data_index.iloc[i]['steering']!=0:
        nozero_set.append(i)
    else:
        zero_set.append(i)
choose=np.random.choice(np.array(zero_set),2000)

nozero_set.extend(list(choose))
       
data_index=old_data_index.iloc[nozero_set]

# s=old_data_index['steering'].values
# plt.hist(s,bins=[-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
# plt.show()
# hist,bins=np.histogram(s,bins=[-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])


class Drive(nn.Module):
 
    def __init__(self, pretrained=True):
        super().__init__()
        self.m=nn.Dropout(p=0.4)
        self.n1=nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5,stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5,stride=2)
        self.conv3 = nn.Conv2d(36,48, kernel_size=5,stride=2)
        self.conv4 = nn.Conv2d(48,64, kernel_size=3)
        self.conv5 = nn.Conv2d(64,64, kernel_size=3)
        
        self.fc1= nn.Linear(64*2*33, 100)
        self.fc2= nn.Linear(100, 50)
        self.fc3= nn.Linear(50, 10)
        
        self.fc4= nn.Linear(10, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
    def forward(self, x):
        x=self.n1(x)
        x = torch.nn.ELU()(self.conv1(x))
        x = torch.nn.ELU()(self.conv2(x))
        x = torch.nn.ELU()(self.conv3(x))
        x = torch.nn.ELU()(self.conv4(x))
        x = torch.nn.ELU()(self.conv5(x))
        x=self.m(x)
        
        # print(x.shape)
        x=x.view(x.size(0), -1)
   
        x=torch.nn.ELU()(self.fc1(x))
        x=self.m(x)
        x=torch.nn.ELU()(self.fc2(x))
        x=torch.nn.ELU()(self.fc3(x))
        
        x=torch.nn.ELU()(self.fc4(x))
        
        
        return x




def read_crop(root):
    pic=cv2.imread(root,cv2.IMREAD_GRAYSCALE)
    pic=pic[192:-90,20:-20]
    pic=cv2.resize(pic,(200,66))
    return pic
    
class my_dataset(Dataset):

    def __init__(self):
       
        self.root=root
       
        self.index=data_index
        
    def __len__(self):
        
           
        return len(self.index)

    def __getitem__(self, idx):
        
         
        r=random.randint(0,2)
        if r==0:
                pic_root=self.index['center'].iloc[idx]
                steering=data_index['steering'].iloc[idx]
        elif r==1:    
                pic_root=self.index['left'].iloc[idx]
                steering=data_index['steering'].iloc[idx]+0.4
        elif r==2:
                pic_root=data_index['right'].iloc[idx]
                steering=self.index['steering'].iloc[idx]-0.4
        # pic_root=data_index['center'].iloc[idx]
        # steering=data_index['steering'].iloc[idx]
        steering+= np.random.normal(loc=0, scale=0.1)
        steering=-1 if steering<-1 else steering
        steering=1 if steering>1 else steering
    
      #  pic_root='/home/xu/Downloads/'+pic_root[8:].replace('\\','/')
        
        pic_root=pic_root.replace('\\','/').replace('D:/jq','/home/xu').replace('D:','/home/xu').replace('Desktop','/home/xu/Downloads')
    
        frame=cv2.imread(pic_root)[55:-35, :, :]
        
        
        
        frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2HSV)
        frame[:, :, 2] = frame[:, :, 2] *random.uniform(0.2, 1.5)
        frame[:, :, 2] = np.clip(frame[:, :, 2], a_min=0, a_max=255)
        pic = cv2.cvtColor(frame, code=cv2.COLOR_HSV2BGR)
        
        
        # trans_x = 320 * (np.random.rand() - 0.5)
        # trans_y = 70 * (np.random.rand() - 0.5)
        # steering+= trans_x * 0.002
        # trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        # height, width = frame.shape[:2]
        # frame = cv2.warpAffine(frame, trans_m, (width, height))
        
        
        pic=np.transpose(pic,(2,0,1))
        
            
        

            
        if np.random.rand() < 0.5 :
            pic = cv2.flip(pic, 1)
            steering = steering * -1.0

        return pic,steering

mydata=my_dataset()

# for i ,j in mydata:
#     print(i)
    
    
    
tsize=int(len(data_index)*0.8)
vsize=(len(data_index)-int(len(data_index)*0.8))
trainset, valset = torch.utils.data.random_split(mydata, [tsize, vsize])




device = torch.device('cuda:0')
model=Drive().to(device)

batch_size=2048
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-5)
train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
test_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=4)



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
total_step=len(data_index)/batch_size
num_epochs=60
train_loss=[]
test_loss=[]
for epoch in range(num_epochs):
    model.train()

    for i,(p,l) in enumerate(train_dataloader):
        optimizer.zero_grad()
        p=p.to(torch.float32).to(device)
        l=l.to(torch.float32).to(device)
        y=model(p).squeeze()
        loss=criterion(y,l)
        loss.backward()
        optimizer.step()
        mloss.add(loss.item())
    train_loss.append(mloss.show())

    print ('Epoch [{}/{}],  train_Loss: {:.4f}' 
                    .format(epoch+1, num_epochs,  train_loss[-1]))
        
    torch.save(model.state_dict(), './saved_model/{}epoch.pth'.format(epoch))
    model.eval()
    #with torch.set_grad_enabled(False):

    for i,(p,l) in enumerate(test_dataloader):
        p=p.to(torch.float32).to(device)
        l=l.to(torch.float32).to(device)
        y=model(p).squeeze()
        # print(y)
        loss=criterion(y,l)
        mloss.add(loss.item())
        
    test_loss.append(mloss.show())
    print ('Epoch [{}/{}],  test_Loss: {:.4f}' 
                    .format(epoch+1, num_epochs,  test_loss[-1]))
    
    if epoch >10 :
        if test_loss[-1]>test_loss[-3] and test_loss[-2]>test_loss[-3] :
            print('early stopping')
            break
        
print('start backup')    
best_epoch=len(test_loss)-3
    
print('show loss')
plt.plot(train_loss)
plt.plot(test_loss)
plt.show()
model.load_state_dict(torch.load('./saved_model/{}epoch.pth'.format(best_epoch)))


x = torch.randn(1, 3, 70, 320, requires_grad=True).to(device)
torch_out = model(x)
torch.onnx.export(model,
                  x,
                  "car_lap1.onnx",
                  export_params = True,
                  opset_version=9,
                  do_constant_folding=True,
                  input_names = ['x'],
                  output_names = ['y']
                  )






















