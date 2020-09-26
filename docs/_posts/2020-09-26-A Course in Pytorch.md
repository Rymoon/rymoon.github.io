---
author: vjy
---

# Pytorch 入门

> 本文翻译自pytorch官方教程[DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZ](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)，共四篇。
> 这是第一篇What is Pytorch?.

教程目标：

* 在整体上理解Pytorch的Tensor(张量)库和神经网络(neural networks)。

* 训练一个小型神经网络来分类图片。

这个教程假设读者对numpy有基本了解。

<!-- more -->

## WHAT IS PYTORCH?什么是 PyTorch?

```python
%matplotlib inline
```


什么是 PyTorch?
================


这是一个针对以下两种需求的基于 Python 科学计算软件包：

-  替代 NumPy 来发挥 GPU 的性能
-  提供最大便捷以及速度的深度学习研究平台

开始
---------------

### Tensors


Tensors 与 NumPy 中的 ndarrays 类似, 另外 Tensors 可以使用 GPU 来加速计算.


```python
from __future__ import print_function
import torch
```

<div class="alert alert-info"><h4>Note</h4><p>当矩阵被声明却未进行初始化，
    在它被使用之前，我们都不能确定矩阵中元素的值. 
    当一个未被初始化的矩阵被创建时,被分配到矩阵的内存
    中存储的值就是矩阵的初始值.</p></div>



创建一个 5x3 的矩阵, 未进行初始化:




```python
x = torch.empty(5, 3)
print(x)
```

创建一个随机初始化的矩阵:




```python
x = torch.rand(5, 3)
print(x)
```

创建一个数据类型为 long 的元素全部为 0 的矩阵:




```python
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
```

直接使用数据创建 tensor:




```python
x = torch.tensor([5.5, 3])
print(x)
```

或者根据已有的 tensor 来创建新的 tensor. 这些方法会反复用到参数的属性， 例如 dtype， 除非值由用户提供.





```python
x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size
```

获得它的大小:


```python
print(x.size())
```

<div class="alert alert-info"><h4>Note</h4><p>``torch.Size`` 实际上是一个 tuple(元组), 所以它支持所有的 tuple 操作.</p></div>

### 运算

针对运算有不同的语法. 在下面的例子中, 我们会涉及加法运算.

加法: 语法 1




```python
y = torch.rand(5, 3)
print(x + y)
```

加法: 语法 2




```python
print(torch.add(x, y))
```

加法: 将运算结果作为参数




```python
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
```

加法: in-place(原地操作符:不经过复制操作，而是直接在原来的内存上改变它的值.)




```python
# adds x to y
y.add_(x)
print(y)
```

<div class="alert alert-info"><h4>Note</h4><p>任何使用 in-place 来改变 tensor 的操作都会在后面加上 ``_``.
    例如: ``x.copy_(y)``, ``x.t_()``, 会改变 ``x``.</p></div>

你可以使用各种标准 NumPy 风格的索引!




```python
print(x[:, 1])
```

调整大小: 如果你想要 resize/reshape(调整大小或改变形状) tensor， 你可以使用``torch.view``:




```python
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())
```

对于只含一个元素的 tensor, 使用 ``.item()`` 来得到它的数值




```python
x = torch.randn(1)
print(x)
print(x.item())
```

**Read later:**

  100+ Tensor 操作, 包括转置， 索引，切片，数学运算， 线性代数， 随机数等等,
  `here <https://pytorch.org/docs/torch>`_.

NumPy Bridge
------------

将 Tensor 转化为 NumPy 中的数组类型或者将数组转化为 Tensor 都是非常简单的.

Torch 中的 tensor 与 NumPy 中的数组会共享底层的内存（如果 tensor 在 CPU 上）， 
且改变其中一个另一个也会改变.

### 将 Tensor 转化为 NumPy 中的数组类型





```python
a = torch.ones(5)
print(a)
```


```python
b = a.numpy()
print(b)
```

NumPy 中的数组如何改变值.




```python
a.add_(1)
print(a)
print(b)
```

### 将 NumPy 的数组转化为 Torch 中的 tensor

数组如何自动转化为 tensor




```python
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
```

除了 CharTensor 所有在 CPU 上的 tensor 都支持在数组之间的转换.

CUDA Tensors
------------
通过使用 ``.to`` Tensors 可以移动到任何设备上.




```python
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
```
