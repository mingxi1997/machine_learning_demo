import torch.nn as nn
import torch
import numpy as np
class AC(nn.Module):  
    def __init__(self,input_shape,n_action):
        super().__init__()
        
        self.conv=nn.Sequential(
            nn.Conv2d(input_shape[0],32,8,4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32,64,4,2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64,64,3,1),
            nn.BatchNorm2d(64),
            nn.ReLU()
            
            )
        conv_out_size = self._get_conv_out(input_shape)
        print('model initialized ,conv_out_size :{} ,output action :{}'.format(conv_out_size,n_action))

        self.actor=nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_action)            
            )
        self.critic=nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)            
            )
        
     
    def forward(self,x):
        out = x.float() / 256
        out=self.conv(out)
        out=out.view(out.size(0),-1)      
        return self.actor(out),self.critic(out)
    
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1,*shape))
        return int(np.prod(o.size()))

class SAC(nn.Module):  
    def __init__(self,):
        super().__init__()
        
        self.actor=nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2)            
            )
        self.critic=nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)            
            )
        
     
    def forward(self,x):
        x = x.float()
        return self.actor(x),self.critic(x)
    

if __name__=="__main__":
    model=SAC()

    x=torch.randn(*(12,4))
    

    y=model(x)
    
  

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    