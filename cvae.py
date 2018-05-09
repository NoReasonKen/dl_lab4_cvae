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
MNIST_PATH="./data"
MNIST_DOWNLOAD=False

TRAIN_DATA_BATCH_SIZE=128
EPOCH=100
LR=1e-3
NOISE_SIZE=20
CONDITION_CODE_SIZE=10
#====================================================================
class CVAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
                nn.Conv2d(11, 3, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(3, 1, kernel_size=3, padding=1),
                nn.ReLU())
        self.fc1 = nn.Sequential(
                nn.Linear(784, 400, True),
                nn.ReLU())
        self.fc21 = nn.Linear(400, 20, True)
        self.fc22 = nn.Linear(400, 20, True)
        self.fc3 = nn.Sequential(
                nn.Linear(30, 392, True),
                nn.ReLU())
        self.conv2 = nn.Sequential(
                nn.Conv2d(2, 11, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(11, 3, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(3, 1, kernel_size=3, padding=1),
                nn.Sigmoid())

    def encoder(self, x, c):
        inputs = torch.cat([x, c], 1)
        inputs = self.fc1(self.conv1(inputs))
        return self.fc21(inputs), self.fc22(inputs)

    def reparameterize(self, u, logv):
        if self.training:
            std = torch.exp(0.5 * logv)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(u)

    def decoder(self, z, c):
        inputs = torch.cat([z, c], 1)
        inputs = self.conv2(self.fc3(inputs))
        return inputs
    
    def forward(self, x, c):
        return self.decoder(self.reparameterize(self.encoder(x, c)), c)
        

#====================================================================
train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(MNIST_PATH, train=True, download=MNIST_DOWNLOAD
                    , transform=transforms.ToTensor()),
        batch_size=TRAIN_DATA_BATCH_SIZE,
        shuffle=True, num_workers=4, 
        )
test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(MNIST_PATH, train=False, download=MNIST_DOWNLOAD
                    , transform=transforms.ToTensor()),
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
    model.train()
    loss = 0
    
    for idx, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()
        data, label = V(data.cuda()), V(label.float().cuda())
        zero = V(torch.zeros(1, 10, 28, 28))
        onehot = V(torch.zeros_like(zero.data))
        for i in range(len(label)):
            tmp = V(torch.zeros(1, 10, 1, 1))
            tmp.data[0][int(label.data[i])][0][0] = 1
            tmp = tmp.expand(1, -1, 28, 28)
            onehot = torch.cat([onehot, tmp], 0)
        onehot = onehot[1:].cuda()

        output, u, logv = model(data, onehot)
        delta = criterion(output, data, u, logv)
        delta.backward()
        optimizer.step()

        loss += delta.data[0]

        progress_bar(idx, len(train_loader), 'Loss: %.3f' % (loss/(idx+1)))

for epoch in range(EPOCH):
    train(epoch)

