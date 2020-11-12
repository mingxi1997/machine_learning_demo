
import cv2  
import numpy
import matplotlib.pyplot as plot
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

def get_imgs(video):
    frame_set=[]
    cap = cv2.VideoCapture(video)
    ret=True

    while True:
    
    # get a frame
        ret, frame = cap.read()
 
    # show a frame
        if ret:
           frame_set.append(frame)
        else:
            break
    
        cv2.waitKey(1)
    cap.release()
    frame_set=np.array(frame_set)
    return frame_set
    


class my_dataset(Dataset):

    def __init__(self):
       self.root = '/home/xu/data/action_rec/'
       
    def __len__(self):
       count=0
       for index in os.listdir(self.root):
           for v in os.listdir(self.root+index):
               count+=1
           
       return count

    def __getitem__(self, idx):
        actions=os.listdir(self.root)
        
        label=np.random.randint(0,len(actions))
        
        action=actions[label]
        
        
        video_name=np.random.choice(os.listdir(self.root+action))
        #print(video_name)

        frame_set=get_imgs(self.root+action+'/'+video_name)

        return frame_set,label







device = torch.device('cuda:0')

# Hyper-parameters
sequence_length = 10
input_size = 500
hidden_size = 300
num_layers = 2
num_classes = 6
batch_size = 1
num_epochs = 10
learning_rate = 1e-4




class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(6,32,3,2,1)
        self.n1=nn.BatchNorm2d(32)
        
        self.c2=nn.Conv2d(32,64,3,2,1)
        self.n2=nn.BatchNorm2d(64)
        
        self.c3=nn.Conv2d(64,64,3,3,1)
        self.n3=nn.BatchNorm2d(64)
    
        self.fc1=nn.Linear(8960,500)

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






class RCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        
        
        self.conv=CNN()
        
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
    
    def forward(self, x):
       
        x=x.permute(0,3,1,2)
        flows=torch.zeros(1,500).to(device)

        
        for i in range(len(x)-1):
            image1=x[i]
            image2=x[i+1]
            image=torch.cat((image1,image2))

            image=image.unsqueeze(0)
            em=self.conv(image)
            
            flows=torch.cat((flows,em))
        flows=flows.unsqueeze(0)
       
 
        
        h0 = torch.zeros(self.num_layers*2, flows.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers*2, flows.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(flows, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

       
        # # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
  
        return out

model = RCNN(input_size, hidden_size, num_layers, num_classes).to(device)

# # Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
my_data=my_dataset()
dataloader = DataLoader(my_data, batch_size=1, shuffle=True, num_workers=1)

total_step=len(my_data)
model.load_state_dict(torch.load('crnn_demo.pt'))
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

for epoch in range(num_epochs):
  for i,(item,label) in enumerate(dataloader):
    # print(item[0].shape,label)
    
    item=item.squeeze().to(torch.float32).to(device)
    label=label.to(device)
    out=model(item)
    loss = criterion(out, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    mloss.add(loss.item())
    if (i+1) % 20 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, mloss.show()))


torch.save(model.state_dict(), 'crnn_demo.pt')

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for item,label in dataloader:
        item=item.squeeze().to(torch.float32).to(device)
        
        label=label.to(device)
        outputs=model(item)
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        print(correct)
        

    print('Test Accuracy of the model on the  test images: {} %'.format(100 * correct / total)) 

















