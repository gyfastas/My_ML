import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from loss_func import *
'''
简单模型，包含3个卷积层和两个全连接层
引入ring loss
'''
class Net(nn.Module):
    def __init__(self,use_Rloss = True):
        super(Net,self).__init__()
        self.conv1 = nn.Sequential(
        nn.Conv2d(1,10,kernel_size=5),
        nn.MaxPool2d(2),
        nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
        nn.Conv2d(10,20,kernel_size=5),
        nn.Dropout2d(0.5),
        nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(320,84)
        self.fc2 = nn.Linear(84,50)
        self.fc3 = nn.Linear(50,10)
        self.ringloss = RingLoss(type = 'auto',loss_weight = 1.0)
        self.softmax = SoftmaxLoss(10,10,normalize = True)
        self.use_Rloss = use_Rloss

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1,320)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x,training=True)
        x = self.fc2(x)
        #x = F.dropout(x,training=True)
        x = self.fc3(x)
        if self.use_Rloss:
            return self.softmax(x,y),self.ringloss(x)
        else:
            return F.log_softmax(x,dim=1)



    def Train(self,args, model, device, train_loader, optimizer, epoch):
        self.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            if self.use_Rloss:
                softmaxloss,ringloss = self(data,target)
                loss = softmaxloss + ringloss
                loss.backward()
            else:
                output = self(data,target)
                loss = F.nll_loss(output,target)
                loss.backward()

            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    def Test(self,args, model, device, test_loader):
        self.eval()
        softmax_loss= 0
        ring_loss = 0
        test_loss = 0
        correct = 0
        with torch.no_grad():
            if self.use_Rloss:
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    sl,rl = self(data,target)  #softmax and ring loss in model
                    softmax_loss += sl  #sum up batch loss
                    ring_loss += rl

                    pred = self.softmax.prob.max(1, keepdim=True)[1] # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

                softmax_loss /= len(test_loader.dataset)
                ring_loss /= len(test_loader.dataset)

                print('\nTest set: Average softmax loss: {:.4f}, Average ring loss: {:.4f} Accuracy: {}/{} ({:.0f}%)\n'.format(
                softmax_loss,ring_loss,correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

            else:
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = self(data,target)
                    test_loss += F.nll_loss(output, target) # sum up batch loss
                    pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

                test_loss /= len(test_loader.dataset)
                print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    test_loss, correct, len(test_loader.dataset),
                    100. * correct / len(test_loader.dataset)))



