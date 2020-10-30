---
author: vjy
catagory: blog
---

# Pytorch 入门

> 本文翻译自pytorch官方教程[DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZ](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)，共四篇。
> 这是第二篇AUTOGRAD: AUTOMATIC DIFFERENTIATION.

<!-- more -->
# Autograd: 自动微分

PyTorch 中的所有神经网络的核心就是 `autograd`包。我们先简单浏览，然后我们将训练我们的第一个神经网络。

`autograd`包为所有 tensors 的操作提供自动微分功能。它是一个在运行式才被定义的框架， 意味着反向传播只有在代码运行时才会计算， 每次迭代时有可能不同。

让我们用一些简单的例子来说明这一点。

## Tensor

`torch.Tensor` 是 PyTorch 中的核心类。 如果你将它的属性 `.requires_grad` 设置为 `True`，它就会开始追踪 在其上的所有操作。当你完成你的计算你可以调用 `backward()`  然后所有的梯度都将自动计算。tensor 的梯度都会被收集到 `.grad` 属性。

要停止 tensor 追踪历史， 你可以调用 `.detach()` ，它将与其计算历史记录分离，并防止将来的计算被追踪。

为了防止追踪历史 (使用内存)，你可以代码块放入 `withtorch.no_grad()`中：这中方法在我们计算一个带有`requires_grad=True` 参数却又不需要梯度的模型时会非常有用。

另外一个对 `autocrat` 实现非常重要的类是 `Function`

`Tensor` 和 `Function` 是相互连接的并且建立了一个记录完整计算历史的无环图。每个 tensor 都拥有 `.grad_fn` 属性保存着创建 tensor 的 Function 的引用，(如果用户自己创建张量，则 `grad_fn` 是 `None`)。

如果你想要计算导数，你可以在 `tensor` 上调用 `.backward()`。如果 `Tensor` 是一个标量 (例如，它只含有一个元素)，你不需要为 `.backward()` 指定任何参数，然而如果它拥有多个元素，则需要指定 `gradient` 参数来指定 tensor 的形状。

```python
import torch
```
创建 tensor 并设置 `requires_grad=True` 来追踪计算
```python3
x = torch.ones(2, 2, requires_grad=True)
print(x)
```
进行 tensor 运算
```python
y = x + 2
print(y)
```
`y` 是运算的结果， 所以它拥有 `grad_fn`.
```python
print(y.grad_fn)
```

对 `y` 进行更多运算
```python
z = y * y * 3
out = z.mean()

print(z, out)
```
`.requires_grad_( ... )` 会改变张量的 `requires_grad` 标记。如果没有提供相应的参数，输入的标记默认为 `False`。

```python
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)
```

## 梯度
我们开始进行反向传播。由于 `out` 包含一个标量， `out.backward()` 与 `out.backward(torch.tensor(1.))` 等价
```python
out.backward()
```
打印梯度 d(out)/dx
```python
print(x.grad)
```
你会得到一个元素全部为 ``4.5`` 的矩阵。我们将 ``out`` 记为 *Tensor* “$o$”。我们得到 $o = \frac{1}{4}\sum_i z_i$,  $z_i = 3(x_i+2)^2$  and  $z_i\bigr\rvert_{x_i=1} = 27$。因此， $\frac{\partial o}{\partial x_i} = \frac{3}{2}(x_i+2)$，$\frac{\partial o}{\partial x_i}\bigr\rvert_{x_i=1} = \frac{9}{2} = 4.5$.

数学上， 如果有一个向量值函数 $\vec{y}=f(\vec{x})$, 那么 $\vec{y}$ 关于 $\vec{x}$ 的梯度是一个雅克比矩阵：

$$
\begin{align}J=\left(\begin{array}{ccc}
   \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\
   \vdots & \ddots & \vdots\\
   \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
   \end{array}\right)\end{align}
$$

一般来说， ``torch.autograd`` 是一个计算 vector-Jacobian product(雅克比矩阵与向量的乘积)的工具。 给定任意向量 $v=\left(\begin{array}{cccc} v_{1} & v_{2} & \cdots v_{m}\end{array}\right)^{T}$, 计算乘积 $v^{T}\cdot J$. 如果 $v$ 恰好是标量函数 $l=g\left(\vec{y}\right)$ 的梯度， 则 $v=\left( \frac{\partial l}{\partial y_{1}} \cdots \frac{\partial l}{\partial y_{m}}\right)^{T}$, 然后通过链式法则，the vector-Jacobian product 就是 $l$ 关于 $\vec{x}$的梯度：
$$
\begin{align}J^{T}\cdot v=\left(\begin{array}{ccc}
   \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{1}}\\
   \vdots & \ddots & \vdots\\
   \frac{\partial y_{1}}{\partial x_{n}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
   \end{array}\right)\left(\begin{array}{c}
   \frac{\partial l}{\partial y_{1}}\\
   \vdots\\
   \frac{\partial l}{\partial y_{m}}
   \end{array}\right)=\left(\begin{array}{c}
   \frac{\partial l}{\partial x_{1}}\\
   \vdots\\
   \frac{\partial l}{\partial x_{n}}
   \end{array}\right)\end{align}
$$

(注意 $v^{T}\cdot J$ 得到一个行向量， 它也可以通过 $J^{T}\cdot v$ 得到一个列向量.)

vector-Jacobian product 的特性使得一个非标量输出的外部梯度更容易被输入模型。

现在我们来看一个 vertor-Jacobian product 的例子
```python
x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)
```

在这个例子中 `y` 不再是标量。`torch.autograd` 不能直接计算完整的雅克比矩阵, 但是如果我们只想要 vector-Jacobian product，只需要将向量传给 `backward` 作为参数。
```python
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)
```
你可以通过将带有 `.requires_grad=True` 的 tensor 放入`with torch.no_grad()`代码块中来防止 autograd 追踪历史：
```python
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
	print((x ** 2).requires_grad)
```
或者通过使用 `.detach()` 来获得一个新的具有同样内容但不需要梯度的 tensor：
```python
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())
```
**Read Later**
有关 `autograd.Function` 的文档 [文档]：https://pytorch.org/docs/stable/autograd.html#function
