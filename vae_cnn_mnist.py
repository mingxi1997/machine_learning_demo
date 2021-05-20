
# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

bs = 3096
# MNIST Dataset
train_dataset = datasets.MNIST(root='/home/xu/py_work/data/mnist', train=True, transform=transforms.ToTensor(), download=False)
test_dataset = datasets.MNIST(root='/home/xu/py_work/data/mnist', train=False, transform=transforms.ToTensor(), download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # encoder part
        
        self.cnn_en=nn.Sequential(
            nn.Conv2d(1, 32, 4, 1, 2),   # B,  32, 28, 28
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  7, 7
        )
        # self.fc1 = nn.Linear(64*7*7, 512)
        # self.fc2 = nn.Linear(512, 256)
        self.fc31 = nn.Linear(64*7*7, 2)
        self.fc32 = nn.Linear(64*7*7, 2)
        # decoder part
        self.fc4 = nn.Linear(2, 64*7*7)
        # self.fc5 = nn.Linear(256, 512)
        # self.fc6 = nn.Linear(512, 784)
        
        self.cnn_de = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  64,  14,  14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1, 1), # B,  32, 28, 28
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 1, 2)   # B, 1, 28, 28
        )
        
        
        
    def encoder(self, x):
     
        h=  torch.relu(self.cnn_en(x))
   
        h=h.view(h.size(0),-1)
 
        # h = torch.relu(self.fc1(h))
        # h = torch.relu(self.fc2(h))
        
        
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
      
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = torch.relu(self.fc4(z)).view(-1, 64, 7, 7)
         
        h=self.cnn_de(h)
        
        return torch.sigmoid(h) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        
        z = self.sampling(mu, log_var)

        return self.decoder(z), mu, log_var

# build model
model = VAE()
if torch.cuda.is_available():
    model.cuda()
    
optimizer = optim.Adam(model.parameters())


for epoch in range(1, 51):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = model(data)
        
        # loss = loss_function(recon_batch, data, mu, log_var)
        BCE = F.binary_cross_entropy(recon_batch, data, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        loss=BCE + KLD 
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    model.eval()
    test_loss= 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.cuda()
            recon, mu, log_var = model(data)
            
            # sum up batch loss
            BCE = F.binary_cross_entropy(recon, data, reduction='sum')
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
            loss=BCE + KLD 
            test_loss += loss.item()
        
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    
    
    
    with torch.no_grad():
        z = torch.randn(64, 2).cuda()
        sample = model.decoder(z).cuda()
    
        save_image(sample.view(64, 1, 28, 28), './samples/{}sample_.png'.format(epoch))    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
