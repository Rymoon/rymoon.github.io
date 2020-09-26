---
author: rym
---
# Pytorch 入门

> 本文翻译自pytorch官方教程[DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZ](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)，共四篇。
> 这是第四篇TRAINING A CLASSIFIER.

<!-- more -->

# Training A Classifier训练分类器


现在，你知道如何定义神经网络，计算损失(loss)，以及更新权重(weight)。现在该考虑一下数据(data)的问题了。

## What about data?

一般来说，处理图片、声音、文本和视频，你可以使用python的标准包来导入，转化为numpy，再变成pytorch中不同种类的张量(torch.*Tensor)。

对于图片，可以用Pillow, OpenCV;
对于音频，可以用scipy和librosa;
对于文本，可以用Python或Cython原生函数或者NLTK和SaCy。

特别对于计算机视觉(vision)，我们已经制作了torchvision包，报扩常用数据集(dataset)如Imagenet, CIFAR10, MNIST等等的加载器(dataloader)和对图像的处理器(transformer)。以上功能通过torchvision.datasets 和torch.utils.data.Dataloader实现。

这提供了巨大的便利，避免重复造轮子。

对于这个教程，我们使用 CIFAR10数据集。它包含以下类别的图片：‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’。其中的图片是3x32x32的，即三通道颜色(RGB)，长宽各32像素(pixel)。

![cifar10]({{site.url}}/\assets\image\2020-09-26-A-Course-in-Pytorch-4\cifar10.png)

## 训练一个分类器=

我们将依次进行以下几个步骤：

* 使用torchvision加载并归一化 CIFAR10的训练集(training dataset)和测试集(test dataset)。
* 定义CNN(卷积神经网络, Convolutional Neural Network)
* 定义损失函数
* 在训练集上训练网络
* 在测试集上测试网络
  
### Loading and normalizing CIFAR10

````python
import torch
import torchvision
import torchvision.transforms as transforms
The output of torchvision datasets are PILImage images of range [0, 1]. We transform them to Tensors of normalized range [-1, 1]. .. note:

If running on Windows and you get a BrokenPipeError, try setting
the num_worker of torch.utils.data.DataLoader() to 0.
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
````

````bat
Out:

Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
Extracting ./data/cifar-10-python.tar.gz to ./data
Files already downloaded and verified
````

让我们显示一些图片：

````python
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
````

![one-pic-in-cifar10]({{site.url}}/\assets\image\2020-09-26-A-Course-in-Pytorch-4\one-pic-in-cifar10.png)

````bat
Out:

ship  bird plane truck
````

### 定义一个CNN

从上一篇文章（神经网络）中复制代码，改动它使能处理3通道图片（之前是1通道）。

````python

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
````

### Define a Loss function and optimizer

用分类的交叉熵来定义损失函数，以及带动量的随机梯度下降法。(Let’s use a Classification Cross-Entropy loss and SGD with momentum.)

````python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
4. Train the network
This is when things start to get interesting. We simply have to loop over our data iterator, and feed the inputs to the network and optimize.

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
````

````bat
Out:

[1,  2000] loss: 2.182
[1,  4000] loss: 1.805
[1,  6000] loss: 1.645
[1,  8000] loss: 1.591
[1, 10000] loss: 1.515
[1, 12000] loss: 1.481
[2,  2000] loss: 1.411
[2,  4000] loss: 1.403
[2,  6000] loss: 1.339
[2,  8000] loss: 1.356
[2, 10000] loss: 1.312
[2, 12000] loss: 1.304
Finished Training
````

保存训练完成的模型。

````python

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
````

更多[关于保存的细节](https://pytorch.org/docs/stable/notes/serialization.html)。

### Test the network on the test data

我们在训练集上训练(train)网络两遍。我们将在测试集上用神经网络预测(predict)图片的分类(label)，和真正分类(ground-truth)对比。如果正确，就把该样本(sample)加入正确预测的列表里。这样就能看出神经网络是否真的学到了什么。

展示测试集的一张图。

````python
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
````

![one-pic-in-test]({{site.url}}/assets/image/2020-09-26-A-Course-in-Pytorch-4/one-pic-in-test.png)

````bat
Out:

GroundTruth:    cat  ship  ship plane
````

我们加载训练好的模型（并非必要，只是展示加载方法）：

````python
net = Net()
net.load_state_dict(torch.load(PATH))
````

现在预测：

````python
outputs = net(images)
````

输出是图片在10类上的能量(机率)，因此取最高作为预测分类：

````python
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

````

````bat
Out:

Predicted:    cat   car plane plane
````

看起来不错。下面在整个测试集上预测。

````python
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
````

````bat
Out:

Accuracy of the network on the 10000 test images: 54 %
````

这看起来比随机选择要好，神经网络确实学到了东西。

下面看看在各个类上的预测表现。

````python
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

````

````bat
Out:

Accuracy of plane : 63 %
Accuracy of   car : 69 %
Accuracy of  bird : 51 %
Accuracy of   cat : 46 %
Accuracy of  deer : 34 %
Accuracy of   dog : 42 %
Accuracy of  frog : 70 %
Accuracy of horse : 53 %
Accuracy of  ship : 57 %
Accuracy of truck : 53 %
````


### Training on GPU

就像把一个张量转移到GPU，我们可以把整个网络转移。

我们先定义GPU设备为第一个可见GPU设备（0号设备），如果cuda可用。

````python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)
````

````bat
Out:

cuda:0
````

之后默认使用这个设备。

这些函数会递归地把所有变量和缓存转移到设备上：

````python

net.to(device)
````

同样，每步的输入和目标也要转移：

````python
inputs, labels = data[0].to(device), data[1].to(device)
````

### 对比CPU，为啥看不出加速效果？

你的网络太小了。

一个练习：增加网络的宽度(width) (第一个nn.Conv2d的参数2,和第二个Conv2d的参数1——它们是相同的数)，看看加速效果。


现在，教程的目的达到了：

* 整体上理解pytorch的张量库和神经网络。

* 训练一个小网络来分类图片。

### 多GPU训练

详见[Data Parallelism](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html).
