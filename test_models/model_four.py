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

# model 3 - wide res net: modified to have 5 mil params

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class BasicBlockWide(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(BasicBlockWide, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        stages = [16, 19*k, 32*k, 70*k]

        self.conv1 = conv3x3(3,stages[0])
        self.layer1 = self._wide_layer(BasicBlockWide, stages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(BasicBlockWide, stages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(BasicBlockWide, stages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(stages[3], momentum=0.9)
        self.linear = nn.Linear(stages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


net=WideResNet(16, 5, 0.3, 10).cuda()

Loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

train_loss_history = []
test_loss_history = []

lambda_val = 0.0001 

for epoch in range(240):
  train_loss = 0.0
  test_loss = 0.0
  for i, data in enumerate(gen([trainDataLoaderTransform,trainDataLoaderSimple])):
    images, labels = data
    images = images.cuda()
    labels = labels.cuda()
    optimizer.zero_grad()
    predicted_output = net(images)
    fit = Loss(predicted_output,labels)

    l2_norm = sum(p.pow(2.0).sum() for p in net.parameters())
    fit += lambda_val * l2_norm

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
  train_loss = train_loss/(2*len(trainingdata_transform))
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