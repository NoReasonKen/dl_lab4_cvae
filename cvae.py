#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable as V
from torchvision import datasets, transforms
from torchvision.utils import save_image

from utils import progress_bar
#===================================================================
MNIST_PATH="../data"
MNIST_DOWNLOAD=True

TRAIN_DATA_BATCH_SIZE=128
EPOCH=100
LR=1e-3
NOISE_SIZE=20
CONDITION_CODE_SIZE=10
#====================================================================
class CVAE(nn.Module):
    def init(self):
        super().__init__()

        self.conv1 = nn.Sequential(
                nn.Conv2d(11, 3, kernel_size=3, padding=1)
                nn.ReLU()
                nn.Conv2d(3, 1, kernel_size=3, padding=1)
                nn.ReLU()
                )
        self.fc1 = nn.Sequential(
                nn.Linear(784, 400, True)
                nn.ReLU()
                )
        self.fc21 = nn.Linear(400, 20, True)
        self.fc22 = nn.Linear(400, 20, True)
        self.fc3 = nn.Sequential(
                nn.Linear(30, 392, True)
                nn.ReLU()
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(2, 11, kernel_size=3, padding=1)
                nn.ReLU()
                nn.UpsamplingNearest2d(scale_factor=2, mode=nearest)
                nn.Conv2d(11, 3, kernel_size=3, padding=1)
                nn.ReLU()
                nn.Conv2d(3, 1, kernel_size=3, padding=1)
                nn.Sigmoid()
                )

    def encoder(self, x, c):
        y = torch.cat([x, c], 1)
        y = self.fc1(self.conv1(y))
        return self.fc21(y), self.fc22(y)

    def reparameterize(self, u, logv):
        if self.training:
            std = torch.exp(0.5 * logv)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(u)
    
    def forward(self, x):
        

#====================================================================
train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(MNIST_PATH, train=True, download=MNIST_DOWNLOAD
                    , transforms=transforms.ToTensor()),
        batch_size=TRAIN_DATA_BATCH_SIZE,
        shuffle=True, num_workers=4, 
        )
test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(MNIST_PATH, train=False, download=MNIST_DOWNLOAD
                    , transforms=transforms.ToTensor()),
        batch_size=1000,
        shuffle=True, num_workers=4, 
        )

cudnn.benchmark = True
model = CVAE().cuda()
optimizer = optim.Adam(model.parameters(), lr=LR)

def criterion(output, data, u, logv):
    MSE = F.mse_loss(output, data.view(-1, 784), size_average=False)
    KLD = -0.5 * torch.sum(1 + logv - u.pow(2) - logv.exp())
    return MSE + KLD

def train(epoch):
    print("Epoch: ", epoch)
    net.train()
    loss = 0
    correct = 0
    total = 0
    
    for idx, (data, label) in enumerate(train_loader):
        optim.zero_grad()
        data, label = V(data.cuda()), V(label.cuda())
        output, u, logv = model(data)
        delta = criterion(output, data, u, logv)
        delta.backward()
        optimizer.step()

        loss += delta.data[0]
        total += label.size(0)
        correct += 

        progress_bar(idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (loss/(idx+1), 100.*correct/total, correct, total))

