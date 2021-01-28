import gym
import random
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
import random
import torch.nn.functional as F


import torchvision.transforms as T


from PIL import Image
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


# torch.manual_seed(1423)
def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)





class NN(nn.Module):  
    def __init__(self):
        super().__init__()
        
        self.c1=nn.Conv2d(3,16,5,2,1)
        self.n1=nn.BatchNorm2d(16)
        
        self.c2=nn.Conv2d(16,32,3,2,1)
        self.n2=nn.BatchNorm2d(32)
        
        self.c3=nn.Conv2d(32,32,3,3,1)
        self.n3=nn.BatchNorm2d(32)
        
        
        
        self.fc1= nn.Linear(1024, 64)
        self.fc3= nn.Linear(64, 2)
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
        
        out=out.view(out.size()[0],-1)
        
        out=self.fc1(out)
     
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

def e_greedy_nn(status,e):
    predict_model.eval()
    # x=torch.tensor(status).to(torch.float32).to(device)
    y=predict_model(status)
    
    action=np.argmax(y.detach().cpu().numpy()) if torch.rand(1, ).item() > e else torch.randint(0, 2,(1,)).item()

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
        env.reset()
    #     experience.append(status)
    # else:
    #     experience.append(status)
        
    experience.append(get_screen())    
    action=e_greedy_nn(get_screen(),e)
    experience.append(action)
    status,reward,done,_=env.step(action)
    experience.append(reward)
    experience.append(get_screen())    
    memory_set.push(experience)
    c+=1

collect_experience(128)
loss_set=[]

def test_time():
  predict_model.eval()
  env.reset()
  done=False
  count=0
    
  while not done:
        count+=1
        #env.render()
        
        action=e_greedy_nn(get_screen(),0)
       
        status, reward, done, info = env.step(action.item())
 
  print(count)
for iteration in tqdm(range(10000)):
    collect_experience(128)
    iteration+=1
    if iteration%10==0:
        test_time()
        x=mloss.show()
        loss_set.append(x)
        print('loss :{}'.format(x))
    if iteration%5==0:
        target_model.load_state_dict(predict_model.state_dict())

    if e>0.05:
        e-=1/5000
    
    for i in range(4):
        sample = random.sample(memory_set.mem, 16)
        s = torch.cat([exp[0] for exp in sample])
        a = torch.tensor([exp[1] for exp in sample]).float().to(device)
        rn = torch.tensor([exp[2] for exp in sample]).float().to(device)
        sn =  torch.cat([exp[0] for exp in sample])

        predict_model.train()

        qp=predict_model(s)
        
        
        
        # predict,_=torch.max(qp, axis=1)
        
        a=a.to(torch.int64).unsqueeze(dim=1)
        predict=qp.gather(1, a).squeeze()
        
        
        
        
        # next_q=torch.max(target_model(sn),axis=1).values*0.95+rn
        
        
        next_a=predict_model(sn).argmax(axis=1).unsqueeze(dim=1)
        
        next_q=target_model(sn).gather(1,next_a).squeeze()
        
        
        

        # target=next_q.detach()
        
      #  print(predict)
   #     print(target)

        loss=criterion(predict,next_q)
        optimizer.zero_grad()
        mloss.add(loss.item())
        loss.backward()
        optimizer.step()
        
     

#torch.save(predict_model.state_dict(), 'our_model.pt')
