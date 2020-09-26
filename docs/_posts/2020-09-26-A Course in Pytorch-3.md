---
author: rym
---
# Pytorch 入门

> 本文翻译自pytorch官方教程[DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZ](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)，共四篇。
> 这是第三篇NEURAL NETWORKS.

<!-- more -->

# 神经网络

用torch.nn搭建神经网络。


nn依赖自动微分(autograd)来搭建模型，并求梯度。一个nn.Module包括网络层(layer)，和（输入到输出的）前向方法(forward)。


举个例子，一个图像分类网络convnet。这是一个简单的前馈网络。输入(input)依次经过各层，到达输出。一个典型的训练过程如下：

![convnet]({{site.url}}/assets/image/2020-09-26-A-Course-in-Pytorch-3/convnet.png)

* 用一些可学习参数定义网络；
* 在输入数据集上迭代；
* 使输入通过网络；
* 计算损失(loss)（输出与正确相差多远）；
* 反向传播梯度(gradient)至各个参数；
* 更新网络参数(weight)，比如：
  `weight = weight - learning_rate * gradient`

## 定义网络

````python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
````

````bat
Out:

Net(
  (conv1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=576, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
````

我们定义了一个前向函数，后向(backward)函数（用于梯度计算）由pytorch的自动微分自动生成。在前向函数可以使用张量运算。

net.parameters()返回模型的可学习参数。

````python
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
````

````bat
Out:

10
torch.Size([6, 1, 3, 3])
````



随机的32x32输入。

>*Note*  
>这个网络(LeNet)的输入应为32x32. 在 MNIST 数据集上使用网络, 需要缩放至32x32.

````python
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
Out:

tensor([[-0.1074, -0.0430,  0.0831, -0.1322,  0.0159, -0.0842,  0.1799,  0.1090,
         -0.0416,  0.0307]], grad_fn=<AddmmBackward>)
````

将所有参数的梯度缓冲区(buffer)置零，并且从随机的梯度开始反向传播。

````python
net.zero_grad()
out.backward(torch.randn(1, 10))

````

>*NOTE*  
>torch.nn只支持mini-batches. torch.nn 包只支持样本的mini-batch作为输入(如：8x32x32)，而不是一个单独的输入（如：32x32）。  
>例如，nn.Conv2d的输入是4维张量nSamples x nChannels x Height x Width。  
>如果只有一个样本，用`input.unsqueeze(0)` 来增加一维(fake batch).

综上，讲到的模块有，

* torch.Tensor - A multi-dimensional array with support for autograd operations like backward(). Also holds the gradient w.r.t. the tensor.
* nn.Module - Neural network module. Convenient way of encapsulating parameters, with helpers for moving them to GPU, exporting, loading, etc.
* nn.Parameter - A kind of Tensor, that is automatically registered as a parameter when assigned as an attribute to a Module.
* autograd.Function - Implements forward and backward definitions of an autograd operation. Every Tensor operation creates at least a single Function node that connects to functions that created a Tensor and encodes its history.

已经完成的部分，

* 定义神经网络
* 处理输入，反向传播

还差，

* 计算损失
* 更新权重(weight)

## 损失函数

损失函数以(output, target) 为输入, 计算一个非负实数来度量输出与目标相差多少。

nn package 有几个不同的损失函数。 nn.MSELoss计算均方误差。

````python
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
````

````bat
Out:

tensor(0.7155, grad_fn=<MseLossBackward>)
Now, if you follow loss in the backward direction, using its .grad_fn attribute, you will see a graph of computations that looks like this:

input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
````

当调用loss.backward()，整个计算图(graph)从loss开始计算微分。计算图中所有属性requires_grad=True的张量(Tensor)将在.grad属性上累加梯度，.grad也是一个张量。


````python
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
````

````bat
Out:

<MseLossBackward object at 0x7fcf8ef75e48>
<AddmmBackward object at 0x7fcf8ef75f60>
<AccumulateGrad object at 0x7fcf8ef75f60>
````

## Backprop

调用loss.backward()开始反向传播，我们需要先清零梯度，否则梯度会累加。

观察反向传播前后conv1.bias的梯度变化。

````python
net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
````

````bat
Out:

conv1.bias.grad before backward
tensor([0., 0., 0., 0., 0., 0.])
conv1.bias.grad after backward
tensor([-0.0038,  0.0054,  0.0083,  0.0059,  0.0141, -0.0297])
````

Now, we have seen how to use loss functions.

>*Read Later * 
>关于neural network package的[更多细节](https://pytorch.org/docs/nn)。

## Update the weights

最简单的更新规则是SGD(随机梯度下降，Stochastic Gradient Descent):

`weight = weight - learning_rate * gradient`

用python实现就是：

````python
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
````

有多种更新方式可用，包括SGD, Nesterov-SGD, Adam, RMSProp,等等。我们通过 torch.optim来使用这些方法。

````python
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
````

>*NOTE*  
>使用 optimizer.zero_grad()将梯度手动清零。这是因为在反向传播中，计算梯度时会累加。
>比如:
>````
>假如
>a := f(x)
>b := g(x)
>y := a+b
>则y.grad := a.grad + b.grad
>````