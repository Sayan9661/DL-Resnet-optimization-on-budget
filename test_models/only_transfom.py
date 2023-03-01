import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt

mean = [0.49139968, 0.48215827 ,0.44653124]
std = [0.24703233, 0.24348505, 0.26158768]
transforms_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.RandomVerticalFlip(p=0.3), 
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomErasing(p=0.75,scale=(0.02, 0.1),value=1.0, inplace=False)
])

transforms_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

 
trainingdata_transform = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_train) 
trainingdata_simple = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_val)

trainDataLoaderTransform = torch.utils.data.DataLoader(trainingdata_transform,batch_size=64,shuffle=True)
trainDataLoaderSimple = torch.utils.data.DataLoader(trainingdata_simple,batch_size=64,shuffle=True)

testdata = torchvision.datasets.CIFAR10(root='./data',  train=False, download=True, transform=transforms_val)
testDataLoader = torch.utils.data.DataLoader(testdata,batch_size=64,shuffle=False)


# combine data loaders
def gen(loaders):
  for loader in loaders:
    for data in loader:
      yield data

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes,fi, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=fi, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=fi,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, num_blocks,c1,p,f=[3,3,3,3]):
        super(ResNet, self).__init__()
        self.in_planes = c1
        self.p = p

        self.conv1 = nn.Conv2d(3, c1, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)

        self.l = len(num_blocks)
        # print(self.l)
        
        self.layer1 = self._make_layer(block, c1, num_blocks[0],f[0], stride=1)

        if self.l>1:
          self.layer2 = self._make_layer(block, 2*c1, num_blocks[1],f[1], stride=2)

        if self.l>2:
          self.layer3 = self._make_layer(block, 4*c1, num_blocks[2],f[2], stride=2)

        if self.l>3:
          self.layer4 = self._make_layer(block, 8*c1, num_blocks[3],f[3], stride=2)

        if self.l>4:
          self.layer5 = self._make_layer(block, 16*c1, num_blocks[4],f[4], stride=2)

        last_in_size = (2**(self.l-1))*c1
        # print('last_in_size',last_in_size)
        last_dim = 64//(2**self.l)
        # print('last_dim',last_dim)
        outsize = (last_dim//(self.p))**2 * last_in_size
        # print('outsize',outsize)
        self.linear = nn.Linear(outsize, 10)

    def _make_layer(self, block, planes, num_blocks,fi, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes,fi,stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        # print('started')
        out = self.layer1(out)
        # print('0')
        if self.l>1:
          out = self.layer2(out)
          # print(out.shape)
        # print('1')
        if self.l>2:
          out = self.layer3(out)
          # print(out.shape)
        # print('2')
        if self.l>3:
          out = self.layer4(out)
          # print(out.shape)
        # print('3')
        if self.l>4:
          out = self.layer5(out)
          # print(out.shape)
        # print('4')
        out = F.avg_pool2d(out, self.p)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.linear(out)
        return out

model = ResNet(BasicBlock,num_blocks = [2,2,2,2],c1 = 42,p = 4).cuda()

net = model
Loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.002)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print('parameters:',count_parameters(model)/1000000,'mil')

train_loss_history = []
test_loss_history = []


for epoch in range(100):
  train_loss = 0.0
  test_loss = 0.0
  for i, data in enumerate(gen([trainDataLoaderTransform,trainDataLoaderSimple])):
    images, labels = data
    images = images.cuda()
    labels = labels.cuda()
    optimizer.zero_grad()
    predicted_output = net(images)
    fit = Loss(predicted_output,labels)


    fit.backward()
    optimizer.step()
    train_loss += fit.item()
  for i, data in enumerate(testDataLoader):
    with torch.no_grad():
      images, labels = data
      images = images.cuda()
      labels = labels.cuda()
      predicted_output = net(images)
      fit = Loss(predicted_output,labels)
      test_loss += fit.item()
  train_loss = train_loss/(2*len(trainingdata_simple))
  test_loss = test_loss/len(testDataLoader)
  train_loss_history.append(train_loss)
  test_loss_history.append(test_loss)
  print('Epoch %s, Train loss %s, Test loss %s'%(epoch, train_loss, test_loss))

print()
print()
print()

correct = 0
total = 0

for i, data in enumerate(testDataLoader):
  with torch.no_grad():
    images, labels = data
    images = images.cuda()
    labels = labels.cuda()
    predicted_output = net(images)
    fit = Loss(predicted_output,labels)
    test_loss += fit.item()
    predicted = torch.max(predicted_output.data, 1)

    for i in range(len(labels)):
      if labels[i] == predicted[1][i]:
        correct += 1
      total += 1

print('accuracy is',correct/total)
