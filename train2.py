import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader





matplotlib.style.use('ggplot')

data_dir = './data/mydata'
data_csv = '/driving_log.csv'
model_json = 'model.json'
model_weights = 'model.h5'
#col_names = ['center', 'left','right','steering','throttle','brake','speed']
# training_dat = pd.read_csv(data_dir+data_csv,names=None)


root = '/home/xu/save/'
# root2 = '/home/xu/save/'
docs = [root+d+'/'+'driving_log.csv' for d in os.listdir(root)]
# docs2 = [root2+d+'/'+'driving_log.csv' for d in os.listdir(root2)]
# docs.extend(docs2)
record = []
for doc in docs:
    record.append(pd.read_csv(doc, names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']))


training_dat = pd.concat(record, axis=0, ignore_index=True)


training_dat.head()

# training_dat[['left','center','right']]
X_train = training_dat[['left','center','right']]
 

Y_train = training_dat['steering']

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

# get rid of the pandas index after shuffling
X_left  = X_train['left'].values
X_right = X_train['right'].values
X_train = X_train['center'].values
X_val   = X_val['center'].values
Y_val   = Y_val.values
Y_train = Y_train.values

Y_train = Y_train.astype(np.float32)
Y_val   = Y_val.astype(np.float32)


def change_root(x):
    return x.replace('\\', '/').replace('D:/jq', '/home/xu').replace('D:', '/home/xu').replace('Desktop', '/home/xu/Downloads')




def read_next_image(m,lcr,X_train,X_left,X_right,Y_train):
    # assume the side cameras are about 1.2 meters off the center and the offset to the left or right 
    # should be be corrected over the next dist meters, calculate the change in steering control
    # using tan(alpha)=alpha

    offset=1.0 
    dist=20.0
    steering = Y_train[m]
    if lcr == 0:
        image = plt.imread(change_root(X_left[m]).strip())
        dsteering = offset/dist * 360/( 2*np.pi) / 25.0
        steering += dsteering
    elif lcr == 1:
        image = plt.imread(change_root(X_train[m]).strip())
    elif lcr == 2:
        image = plt.imread(change_root(X_right[m]).strip())
        dsteering = -offset/dist * 360/( 2*np.pi)  / 25.0
        steering += dsteering
    else:
        print ('Invalid lcr value :',lcr )
    
    return image,steering

def random_crop(image,steering=0.0,tx_lower=-20,tx_upper=20,ty_lower=-2,ty_upper=2,rand=True):
    # we will randomly crop subsections of the image and use them as our data set.
    # also the input to the network will need to be cropped, but of course not randomly and centered.
    shape = image.shape
    col_start,col_end =abs(tx_lower),shape[1]-tx_upper
    horizon=60;
    bonnet=136
    if rand:
        tx= np.random.randint(tx_lower,tx_upper+1)
        ty= np.random.randint(ty_lower,ty_upper+1)
    else:
        tx,ty=0,0
    
    #    print('tx = ',tx,'ty = ',ty)
    random_crop = image[horizon+ty:bonnet+ty,col_start+tx:col_end+tx,:]
    image = cv2.resize(random_crop,(64,64),cv2.INTER_AREA)
    # the steering variable needs to be updated to counteract the shift 
    if tx_lower != tx_upper:
        dsteering = -tx/(tx_upper-tx_lower)/3.0
    else:
        dsteering = 0
    steering += dsteering
    
    return image,steering

def random_shear(image,steering,shear_range):
    rows,cols,ch = image.shape
    dx = np.random.randint(-shear_range,shear_range+1)
    #    print('dx',dx)
    random_point = [cols/2+dx,rows/2]
    pts1 = np.float32([[0,rows],[cols,rows],[cols/2,rows/2]])
    pts2 = np.float32([[0,rows],[cols,rows],random_point])
    dsteering = dx/(rows/2) * 360/(2*np.pi*25.0) / 6.0    
    M = cv2.getAffineTransform(pts1,pts2)
    image = cv2.warpAffine(image,M,(cols,rows),borderMode=1)
    steering +=dsteering
    
    return image,steering

def random_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = 0.8 + 0.4*(2*np.random.uniform()-1.0)    
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def random_flip(image,steering):
    coin=np.random.randint(0,2)
    if coin==0:
        image,steering=cv2.flip(image,1),-steering
    return image,steering
        


def get_validation_set(X_val,Y_val):
    X = np.zeros((len(X_val),64,64,3))
    Y = np.zeros(len(X_val))
    for i in range(len(X_val)):
        x,y = read_next_image(i,1,X_val,X_val,X_val,Y_val)
        X[i],Y[i] = random_crop(x,y,tx_lower=0,tx_upper=0,ty_lower=0,ty_upper=0)
    return X,Y
    




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





class Drive(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.dp = nn.Dropout(p=0.5)
        # self.n1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 24, kernel_size=8, stride=4,padding=2)
        self.conv2 = nn.Conv2d(24, 64, kernel_size=8, stride=4,padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2,padding=2)
        
        # self.conv1 = Conv2d(3, 24, kernel_size=8, stride=4,padding=2)
        # self.conv2 = Conv2d(24, 64, kernel_size=8, stride=4,padding=2)
        # self.conv3 = Conv2d(64, 128, kernel_size=4, stride=2,padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=2, stride=1)
     

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

 
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # x = self.n1(x)
        x = torch.relu((self.conv1(x)))
        x = torch.relu((self.conv2(x)))
        x = torch.relu((self.conv3(x)))
        x = torch.relu((self.conv4(x)))
        # # # 
        x = x.reshape(x.size(0), -1)
        x = self.dp(x)
        x= torch.relu(self.fc1(x))
        x = self.dp(x)
        x= self.fc2(x)
        x= self.fc3(x)
        # x=torch.tanh(x)
        
    
        return x




class my_dataset(Dataset):

    def __init__(self):
       
        self.root=X_train
        self.X_left=X_left
        self.X_right=X_right
        self.Y_train=Y_train
        self.X_train=X_train
       
    def __len__(self):
        
           
        return len(self.Y_train)

    def __getitem__(self, idx):
        
        
        # m = np.random.randint(0,len(Y_train))
#    print('training example m :',m)
        lcr = np.random.randint(0,3)
    #lcr = 1
#    print('left_center_right  :',lcr)
        image,steering = read_next_image(idx,lcr,self.X_train,self.X_left,self.X_right,self.Y_train)
#    print('steering :',steering)
#    plt.imshow(image)
        image,steering = random_shear(image,steering,shear_range=100)
#    print('steering :',steering)
#    plt.figure()
#    plt.imshow(image)    
        image,steering = random_crop(image,steering,tx_lower=-20,tx_upper=20,ty_lower=-10,ty_upper=10)
#    print('steering :',steering)
#    plt.figure()
#    plt.imshow(image)
        image,steering = random_flip(image,steering)
#    print('steering :',steering)
#    plt.figure()
#    plt.imshow(image)
    
        image = random_brightness(image)
        
        image=image/127.5-1.0
        
        image=image.transpose(2,0,1)
#    plt.figure()
#    plt.imshow(image)
    
        return image,steering
       
class val_dataset(Dataset):

    def __init__(self):
       
     
        self.x=X_val
        self.y=Y_val
       
    def __len__(self):
        
           
        return len(self.y)

    def __getitem__(self, idx):
        
        
        X = np.zeros((len(X_val),64,64,3))
        Y = np.zeros(len(X_val))
       
        x,y = read_next_image(idx,1,self.x,self.x,self.x,self.y)
        X,Y = random_crop(x,y,tx_lower=0,tx_upper=0,ty_lower=0,ty_upper=0)
        X=X/217.5-1.0
        
        X=X.transpose(2,0,1)
#    plt.imshow(image)
    
        return X,Y  
       
        
       
        

mydata= my_dataset()
valdata=val_dataset()
batch_size=1024
train_dataloader = DataLoader(mydata, batch_size=batch_size, shuffle=False, num_workers=4)

val_dataloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, num_workers=4)



X_val,Y_val = get_validation_set(X_val,Y_val)
criterion = nn.MSELoss()
    
device = torch.device('cuda:0')
model = Drive().to(device)
    
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,weight_decay=0.0)

train_loss=[]
test_loss=[]

epoch=0
batch_size=1024
iterations=int(len(X_train)/batch_size)+1

num_epochs=20


for epoch in range(num_epochs):
    model.train()
    for i,(p ,l) in enumerate(train_dataloader):
      optimizer.zero_grad()
      p = p.to(torch.float32) .to(device)

      l = l.to(torch.float32).to(device)
      
      y=model(p).squeeze()
      
      loss=criterion(y,l)
      loss.backward()
      optimizer.step()
      mloss.add(loss.item()) 
      
      
    train_loss.append(mloss.show())
      
      
           
    print ('Epoch [{}/{}],  train_Loss: {:.4f}' 
                    .format(epoch+1, num_epochs,  train_loss[-1]))
          
      
    torch.save(model.state_dict(), './model/{}epoch.pth'.format(epoch))
    
    model.eval()
    for vp ,vl in val_dataloader:
     
              vp = vp.to(torch.float32) .to(device)
              vl = vl.to(torch.float32).to(device)
              vy=model(vp).squeeze()
    
              vloss=criterion(vy,vl)
   
              nloss.add(vloss.item()) 
      
    test_loss.append(nloss.show())
      
      
    print ('Epoch [{}/{}],  test_Loss: {:.4f}' 
                    .format(epoch+1, num_epochs,  test_loss[-1]))
    if epoch>4:
        if test_loss[-1]>test_loss[-2] and test_loss[-2]>test_loss[-3]:
            print('early stopping')
            break      
        
best_epoch=len(test_loss)-3    
model.load_state_dict(torch.load('./model/{}epoch.pth'.format(best_epoch)))
plt.plot(train_loss)
plt.plot(test_loss)
plt.show()    
    
    
x = torch.randn(1, 3, 64, 64, requires_grad=True).to(device)
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

