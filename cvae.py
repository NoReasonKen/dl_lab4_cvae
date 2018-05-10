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

import matplotlib.pyplot as plt

from utils import progress_bar
#===================================================================
MNIST_PATH="./data"
MNIST_DOWNLOAD=False

TRAIN_DATA_BATCH_SIZE=128
EPOCH=100
LR=1e-3
NOISE_SIZE=20
CONDITION_CODE_SIZE=10

MODE="load"
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
                nn.Upsample(scale_factor=2),
                nn.Conv2d(11, 3, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(3, 1, kernel_size=3, padding=1),
                nn.Sigmoid())

    def encoder(self, x, c):
        onehot = V(torch.zeros(1, 10, 28, 28)).cuda()
        for i in range(len(c)):
            tmp = V(torch.zeros(1, 10, 1, 1)).cuda()
            tmp.data[0][int(c.data[i])][0][0] = 1
            tmp = tmp.expand(1, -1, 28, 28)
            onehot = torch.cat([onehot, tmp], 0)
        onehot = onehot[1:]
        
        inputs = torch.cat([x, onehot], 1)
        inputs = self.fc1(self.conv1(inputs).view(-1, 1, 784))
        return self.fc21(inputs), self.fc22(inputs)

    def reparameterize(self, u, logv):
        if self.training:
            std = torch.exp(0.5 * logv)
            eps = V(torch.randn(len(u), 1, 20)).cuda()
            return eps.mul(std).add_(u)

    def decoder(self, z, c):
        onehot = V(torch.zeros(len(c), 1, 10)).cuda()
        for i in range(len(c)):
            onehot.data[i][0][int(c.data[i])] = 1
        inputs = torch.cat([z, onehot], 2)
        inputs = self.conv2(self.fc3(inputs).view(-1, 2, 14, 14))
        return inputs
    
    def forward(self, x, c):
        u, logv = self.encoder(x, c)
        return self.decoder(self.reparameterize(u, logv), c), u, logv
        

#====================================================================
def criterion(output, data, u, logv):
    MSE = F.mse_loss(output, data, size_average=False)
    KLD = -0.5 * torch.sum(1 + logv - u.pow(2) - logv.exp())
    return MSE + KLD

def train(epoch):
    print("Epoch: ", epoch)
    model.train()
    loss = 0
    
    for idx, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()
        data, label = V(data.cuda()), V(label.float().cuda())

        output, u, logv = model(data, label)
        delta = criterion(output, data, u, logv)
        delta.backward()
        optimizer.step()

        loss += delta.data[0]

        progress_bar(idx, len(train_loader), 'Loss: %.3f' % (loss/(idx+1)))
    return loss

def save_model():
    model = CVAE().cuda()
    lr = LR
    optimizer = optim.Adam(model.parameters(), lr=lr)

    thredhold = 5
    loss = 0
    loss_log = 1e10
    for epoch in range(EPOCH):
        loss = train(epoch)
        
        if (loss_log - loss) / 469 < thredhold:
            lr *= 0.9
            thredhold *= 0.99
            print(lr, thredhold)
            optimizer = optim.Adam(model.parameters(), lr=lr)

        loss_log = loss

    torch.save(model, 'model.pkl')

def load_model():
    model = torch.load('model.pkl').cuda()

    noise = torch.randn(1, 1, 20).expand(10, 1, 20)
    for i in range(9):
        tmp = torch.randn(1, 1, 20).expand(10, 1, 20)
        noise = torch.cat([noise, tmp], 0)
    noise = V(noise.cuda())
    cond = [torch.arange(10)] * 10
    cond = V(torch.cat(cond, 0).cuda())

    sample = model.decoder(noise, cond).cpu()

    save_image(sample.data, 'output.jpg', nrow=10)

if __name__ == '__main__':
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(MNIST_PATH, train=True, download=MNIST_DOWNLOAD
                        , transform=transforms.ToTensor()),
            batch_size=TRAIN_DATA_BATCH_SIZE,
            shuffle=True, num_workers=4, 
            )
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(MNIST_PATH, train=False, download=MNIST_DOWNLOAD
                        , transform=transforms.ToTensor()),
            batch_size=100,
            shuffle=True, num_workers=4, 
            )

    cudnn.benchmark = True

    if MODE == 'load':
        load_model()
    else:
        save_model()
