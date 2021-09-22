import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

############################
# edit this for amp
from apex import amp
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel
############################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output



def train(args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(data)
        loss = F.nll_loss(output, target)

	
        #################################################
        # edit this for amp
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # loss.backward()
        #################################################
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))







parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', default=True,
                    help='For Saving the current Model')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()


dist.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)



torch.manual_seed(args.seed)
train_kwargs = {'batch_size': args.batch_size}
test_kwargs = {'batch_size': args.test_batch_size}
cuda_kwargs = {'num_workers': 1,'pin_memory': True,'shuffle': False}
train_kwargs.update(cuda_kwargs)
test_kwargs.update(cuda_kwargs)
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
dataset1 = datasets.MNIST('./ms', train=True, download=True,transform=transform)
dataset2 = datasets.MNIST('./ms', train=False, transform=transform)

train_sampler = torch.utils.data.distributed.DistributedSampler(dataset1)
test_sampler = torch.utils.data.distributed.DistributedSampler(dataset2)
train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset2, **train_kwargs, sampler=test_sampler)



model = Net().cuda()
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)


print("="*100)
print(model)

##############################################################################################3
#add this for amp
opt_level = 'O1'
model = convert_syncbn_model(model).cuda()
model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level,loss_scale=128.0)
model = DistributedDataParallel(model)
###############################################################################################

scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
for epoch in range(1, args.epochs + 1):
    train(args, model, train_loader, optimizer, epoch)
    test(model, test_loader)
    scheduler.step()
if args.save_model:
    torch.save(model.state_dict(), "mnist_cnn.pt")
    # torch.save(model, "mnist_cnn.pt")  会报错，只能保存模型参数，不能保存模型

CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 mnist.py
