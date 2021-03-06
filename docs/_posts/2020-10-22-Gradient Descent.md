---
author: stg1
catagory: blog
---
# *机器学习导论* 算法及其实现 1
## 作者：崔嘉珩
>在本章中，我们将重点介绍优化方法中的梯度下降，并说明梯度下降在模型调优中如何工作。 最后，我们将介绍一些改进的梯度下降方法。
<!-- more -->

## 1.梯度下降法
### (1) 基本思想
梯度下降法是一种一阶最优化算法，用来寻找一个可微函数的局部最小值.

我们想要解决如下的最优化问题：
\\[ \mathop{\min}\limits_{x \in R^n} f(x) \\]

令$f(x)$为一可微函数，$ x^{*} $是$f(x)$的局部最小值点.

我们选择一个起始点$x_0$并用下面的迭代公式来进行迭代:
$$ x_{k+1} = x_k + step_k p_k, k = 0,1,2,... $$

其中$step_k$是第$k$步的步长（也称为学习率），$p_k$是第$k$步的下降方向.

由于$f(x)$ 在$x_k$点沿**梯度的负方向** $-\nabla f(x_k)$使函数值下降最快，我们令第$k$步的下降方向$p_k = -\nabla f(x_k)$.

之后我们可以通过线性搜索来解出$step_k$，即$step_k = \mathop{\arg\min}\limits_{step_k \ge 0} f(x_k + step_k p_k)$. 我们也可以选择用固定的步长，步长的选择一般因问题而定.

注意到如果$f(x)$是凸函数（指下凸），那么极小值点只有一个,即$x^{ * }$，其也是全局最小值点. 则我们用梯度下降得到的极小值点就是 $x^{ * }$（若不考虑算法带来的误差）.
### (2) 二次函数情况
若$f(x)$是正定二次函数, 即
$$f(x) = \frac{1}{2}x^T A x + b^T x + c, A \gt 0$$

其中$A \gt 0$指$A$是${n \times n}$正定矩阵，$x$是一个${n \times 1}$向量，则$f(x)$在$R^{n \times 1}$上是凸函数.

令$g_k = \nabla f(x_k) = Ax_k + b$，则下降方向$p_k = -g_k$，此时迭代公式为$x_{k+1} = x_k - step_k g_k$.

此时我们可以推导$step_k$的显式表示：$step_k$会让$f(x)$在$g_k$方向上达到最小，则$step_k$应该能使$\phi(\alpha) = f(x_k - \alpha g_k)$达到最小值.

我们有$\phi'(\alpha) = ((x_k - \alpha g_k)^T A + b^T) (-g_k) $和$\phi'(step_k) = 0$，则能求出$step_k = \frac{g_k^T g_k}{g_k^T A g_k}$. 

此时我们可以得到最终的迭代公式：$$x_{k+1} = x_k - \frac{g_k^T g_k}{g_k^T A g_k} g_k, g_k = Ax_k + b$$

### (3) 算法
输入：目标函数$f(x)$，梯度函数$g(x) = \nabla f(x)$，精度$\epsilon$，起始点$x_0 \in R^{n \times 1}$，最大循环次数$k_{max}$.

输出：$f(x)$的一个极小值点$x^{*}$.

第一步：载入$x_0$，并令$k = 0$.

第二步：计算$f(x_k)$

第三步：计算$g_k = g(x_k)$，若$\parallel g_k\parallel \lt \epsilon$，停止循环并令$x^* = x_k$；否则，令$p_k = -g_k$并计算$step_k = \mathop{\arg\min}\limits_{step_k \ge 0} f(x_k + step_k p_k)$.

**注：在本章中，$\parallel \parallel$指的是$L^2$范数.**

第四步：令$x_{k+1} = x_k + step_k p_k$，计算$f(x_{k+1})$.

第五步：若$\parallel f(x_{k+1}) - f(x_{k})\parallel \lt \epsilon$或$\parallel x_{k+1} - x_{k}\parallel \lt \epsilon$，停止循环，并令$x^* = x_{k+1}$；否则，令$k = k+1$，**回到第三步**.

第六步：若$k = k_{max}$，循环不收敛，则输出"循环不收敛，计算失败.".

### (4) 代码


```python
import numpy as np
```

一般情况下，解$step_k$很麻烦. 我们讨论两种简单情况：固定的学习率和二次函数情况.

<1> 用户事先给出给定的学习率$\eta$，则$step_k = \eta, k=0,1,2,...$：


```python
def gradient_descent_fixed_eta(f_x, g_x, eps, x_0, eta, k_max):
    x_k = x_0
    k = 0
    
    for k in range(k_max+1):#range(n) 会输出如下序列：0, 1, ..., n-1
        g_k = g_x(x_k)
        
        if(np.linalg.norm(g_k) < eps):#np.linalg.norm(g_k)是g_k的L2范数
            return (x_k,k)
            break
        
        else:
            p_k = - g_k
            x_k_new = x_k + eta * p_k
            
            if(np.linalg.norm(f_x(x_k_new) - f_x(x_k)) < eps or np.linalg.norm(x_k_new - x_k) < eps):
                return (x_k_new,k)
                break
                
            else:
                x_k = x_k_new
                if(k==k_max):
                    return "循环不收敛，计算失败."
```

<2> 二次函数情况，此时用户应给出$A$：


```python
def gradient_descent_quadratic(f_x, g_x, eps, x_0, A, k_max):
    x_k = x_0
    k = 0
    
    for k in range(k_max+1):
        g_k = g_x(x_k)
        
        if(np.linalg.norm(g_k) < eps):
            return (x_k,k)
            break
            
        else:
            step_k = g_k.T.dot(g_k)/g_k.T.dot(A).dot(g_k)#.T 指矩阵转置， .dot() 指矩阵乘法
            x_k_new = x_k - step_k * g_k
            
            if(np.linalg.norm(f_x(x_k_new) - f_x(x_k)) < eps or np.linalg.norm(x_k_new - x_k) < eps):
                return (x_k_new,k)
                break
                
            else:
                x_k = x_k_new
                if(k==k_max):
                    return "循环不收敛，计算失败."
```

我们来给出一个简单的例子：令$f(x_1, x_2) = x_1^2 + 4x_2^2$. 明显$f(x)$是凸函数，且唯一的极小值点是$(0,0)$.

我们将$f(x_1, x_2)$变为矩阵形式：$f(x) = \frac{1}{2}x^T Ax$，其中$x = (x_1,x_2)^T, A = \left\[\begin{matrix} 2 & 0\\ 0 & 8\\ \end{matrix}\right\] $. 则$g(x) = \nabla f(x) = Ax$.


```python
A = np.array([[2,0], [0,8]])
x_0 = np.array([1,1])

def f_test(x):
    return 1/2 * x.T.dot(A).dot(x)

def g_test(x):
    return A.dot(x)

print(gradient_descent_fixed_eta(f_test, g_test, 1e-5, x_0, 0.1, 1000))
print(gradient_descent_quadratic(f_test, g_test, 1e-5, x_0, A, 1000))
```

    (array([3.77789319e-03, 3.35544320e-18]), 24)
    (array([ 1.00365696e-03, -6.27285599e-05]), 6)


我们可以看到，若$f(x)$是二次函数，最好用$step_k$的显式公式. 用了$step_k$显式公式的循环次数是6，使用固定学习率的则是24.

### (5) 在机器学习中，可以用梯度下降来做什么？

在机器学习任务中，我们有训练集$S = {(x_1,y_1),(x_2,y_2),...,(x_n,y_n)}$，其中$x_i \in R^{p \times 1} ,y_i \in R$，还有一个模型，其中的参数向量为$\theta = (\theta_1, \theta_2, ... ,\theta_p)^T \in R^{p \times 1}$. 我们训练的目标是最小化损失函数$L(\theta) = \frac{1}{n}\sum_{i=1}^{n}L(f(x_i,\theta),y_i)$.

假设我们想训练一个线性模型，并用MSE作为我们的损失函数：
$$\begin{aligned} MSE(\theta) = \frac{1}{n} \parallel X \theta - y\parallel^2= \frac{1}{n}\sum_{i=1}^{n}(\theta^T x_i - y_i)^2
\end{aligned}$$

其中$X = (x_1, x_2, ..., x_n)^T \in R^{n \times p}, y = (y_1, y_2, ..., y_n)^T \in R^{n \times 1}$.

如果我们想用梯度下降来最小化MSE，我们要先计算它的梯度：
$$\nabla MSE(\theta) = \frac{2}{n} X^T(X\theta-y)$$

则迭代公式为$\theta_{k+1} = \theta_k - step_k \nabla MSE(\theta)$.

我们可以利用python代码来找到使模型最优的$\theta$. 我们使用一个用户给定的学习率$\eta$:


```python
def MSE(theta, X, y):
    n = X.shape[0]
    return 1/n * np.linalg.norm(X.dot(theta)-y)^2

def MSE_gradient(theta, X, y):
    n = X.shape[0]
    return 2/n * X^T.dot(X.dot(theta)-y)

def MSE_gradient_descent_fixed_eta(theta_0, X, y, eta, eps, k_max):
    theta_k = theta_0
    k = 0
    
    for k in range(k_max+1):
        g_k = MSE_gradient(theta_k, X, y)
        
        if(np.linalg.norm(g_k) < eps):
            return (theta_k,k)
            break
        
        else:
            theta_k_new = theta_k - eta * g_k
            
            if(np.linalg.norm(MSE(theta_k_new, X, y) - MSE(theta_k, X, y)) < eps or np.linalg.norm(theta_k_new - theta_k) < eps):
                return (theta_k_new,k)
                break
                
            else:
                theta_k = theta_k_new
                if(k==k_max):
                    return "循环不收敛，计算失败."
```

## 2.共轭梯度下降
### (1) 二次函数
当$x_k$贴近$x^*$时，尤其是当$f(x)$较为复杂时，收敛速度将会越来越慢. 此时共轭梯度下降将会表现得比梯度下降要好.

共轭的定义：令$A$为一个正定矩阵，$Q_1$和$Q_2$是两个非零向量，则$Q_1$和$Q_2$关于$A$共轭，若$Q_1^T A Q_2 = 0$.

我们继续考虑二次函数的情况：$$\begin{aligned} \mathop{\min}\limits_{x \in R^n} f(x) = \mathop{\min}\limits_{x \in R^n}\frac{1}{2}x^T A x + b^T x + c, A \gt 0 \end{aligned}$$

现在的迭代公式为$x_{k+1} = x_k + \alpha_k p_k$.

梯度下降中$p_{k+1} = -\nabla f(x_{k+1})$，现在我们利用$step_k$和$p_k$来修正 $p_{k+1}$：
$$\begin{aligned} p_{k+1} =  -\nabla f(x_{k+1}) + step_k p_k, p_0 = -\nabla f(x_0) \end{aligned}$$

令$p_{k+1}$和$p_k$共轭，则$0 = p_{k+1}^T A p_k = -\nabla f(x_{k+1})A p_k + step_k p_k^TA p_k$. 则可以算出$step_k = \frac{-\nabla f(x_{k+1})^T A p_k}{p_k^TA p_k}$.

利用线性搜索，我们可以得到$\alpha_k = \frac{p_k^T(-b-Ax_k)}{p_k^T A p_k}$.

最终，每次迭代我们都先要更新$p_k = - \nabla f(x_k) - \frac{\nabla f(x_{k})^T A p_{k-1}}{p_{k-1}^TA p_{k-1}}p_{k-1}$. 再用已经更新的$p_k$计算$\alpha_k = \frac{p_k^T(-b-Ax_k)}{p_k^T A p_k}$. 最终$x_{k+1} = x_k + \alpha_k p_k$.

### (2) 算法
输入：目标函数$f(x)$，梯度函数$g(x) = \nabla f(x)$，精度$\epsilon$，起始点$x_0 \in R^{n \times 1}$，在$f(x)$中出现的正定阵$A \in R^{n \times n}$和$b \in R^{n \times 1}$，最大迭代次数$k_{max}$.

输出：$f(x)$的一个极小值点$x^{*}$.

第一步：载入$x_0$，并令$k = 0$.

第二步：计算$f(x_k)$

第三步：计算$g_k = g(x_k)$，若$\parallel g_k\parallel \lt \epsilon$，停止循环并令$x^* = x_k$；否则，若$k = 0$，令$p_k = -g_k$，若$k > 0$, 则令$p_k = -g_k  - \frac{g_k^T A p_{k-1}}{p_{k-1}^TA p_{k-1}}p_{k-1}$并计算$\alpha_k = \frac{p_k^T(-b-Ax_k)}{p_k^T A p_k}$.

第四步：令$x_{k+1} = x_k + step_k p_k$，计算$f(x_{k+1})$.

第五步：若$\parallel f(x_{k+1}) - f(x_{k})\parallel \lt \epsilon$或$\parallel x_{k+1} - x_{k}\parallel \lt \epsilon$，停止循环，并令$x^* = x_{k+1}$；否则，令$k = k+1$，**回到第三步**.

第六步：若$k = k_{max}$，循环不收敛，则输出"循环不收敛，计算失败.".

### (3) Code


```python
def conjugate_gradient_descent_quadratic(f_x, g_x, eps, x_0, A, b, k_max):
    x_k = x_0
    k = 0
    p_k_old = 0 * x_0
    
    for k in range(k_max+1):
        g_k = g_x(x_k)
        
        if(np.linalg.norm(g_k) < eps):
            return (x_k,k)
            break
            
        elif(k == 0):
            p_k = -g_k
            
        elif(k > 0):
            p_k = -g_k - g_k.T.dot(A).dot(p_k_old)/p_k_old.T.dot(A).dot(p_k_old) * p_k_old
            
        alpha_k = p_k.T.dot(-b-A.dot(x_k))/p_k.T.dot(A).dot(p_k)
        x_k_new = x_k + alpha_k * p_k
            
        if(np.linalg.norm(f_x(x_k_new) - f_x(x_k)) < eps or np.linalg.norm(x_k_new - x_k) < eps):
            return (x_k_new,k)
            break
                
        else:
            x_k = x_k_new
            p_k_old = p_k
            if(k==k_max):
                return "循环不收敛，计算失败."
```

我们给出一个简单的例子：$f(x_1, x_2) = x_1^2 - 2x_1 + 4x_2^2 - 16x_2$. 则明显$f(x)$是凸函数，最小值点是$(1,2)$.

将$f(x_1, x_2)$写成矩阵形式：$f(x) = \frac{1}{2}x^T Ax + b^Tx$，其中$x = (x_1,x_2)^T, A = \left[\begin{matrix} 2 & 0\\ 0 & 8\\ \end{matrix}\right], b = (-2, -16)^T $. 则$g(x) = \nabla f(x) = Ax + b$.


```python
A = np.array([[2,0], [0,8]])
b = np.array([-2,-16])
x_0 = np.array([0,0])

def f_test(x):
    return 1/2 * x.T.dot(A).dot(x) + b.T.dot(x)

def g_test(x):
    return A.dot(x) + b

print(conjugate_gradient_descent_quadratic(f_test, g_test, 1e-5, x_0, A, b, 1000))
```

    (array([0.99859738, 1.99989539]), 10)


## 3.随机梯度下降
### (1) 基本思想
在第一部分计算损失函数值时，我们用到了所有的样本$x_1, x_2, ..., x_n$. 因此，当$n$非常大时，计算损失函数值和梯度将会非常麻烦.

现在，我们介绍随机梯度下降法，该方法仅使用一个随机样本来计算损失函数及其梯度. 当处理巨大的训练集时，这将使算法更快；并且随机梯度下降可能使迭代跳出局部最小值的邻域，并找到全局最小值. 但是，当最终接近最小值时，迭代结果将上下浮动，因此我们永远找不到最佳参数，而只能找到一个较好的参数.

我们继续考虑MSE情况，每次循环我们都从$X$中随机抽取一个$x_i$，并从$y$中找到其对应的$y_i$，然后放入如下的公式中：

$$  MSE(\theta | x_i, y_i) = (\theta^T x_i - y_i)^2 $$

我们可以计算出梯度：

$$\nabla MSE(\theta | x_i, y_i) = 2(x_i^T\theta-y_i) x_i .$$

现在迭代公式为$\theta_{k+1} = \theta_k - step_k \nabla MSE(\theta | x_i, y_i)$. 这里我们使用一个简单的学习计划：$step_k = \frac{5}{n+k+50}$, 这会让步长越来越短，并且抑制$\theta$来回波动.
### (2) 代码


```python
def MSE_one_sample(theta, x_i, y_i):
    return np.linalg.norm(theta.T.dot(x_i)-y_i)^2

def MSE_gradient_one_sample(theta, x_i, y_i):
    return 2 * (x_i^T.dot(theta)-y_i) * x_i

def learning_schedule(t):
    return 5/(t + 50)

def MSE_stochastic_gradient_descent(theta_0, X, y, eta, eps, k_max):
    theta_k = theta_0
    k = 0
    n = X.shape[0]
    
    for k in range(k_max+1):
        i = np.random.randint(1, n+1)#randint(1, n+1) 会从{1,2,...,n}中随机选出一个整数
        g_k = MSE_gradient_one_sample(theta_k, X[i], y[i])
        
        if(np.linalg.norm(g_k) < eps):
            return (theta_k,k)
            break
        
        else:
            eta = learning_schedule(n + k)
            theta_k_new = theta_k - eta * g_k
            
            if(np.linalg.norm(MSE_one_sample(theta_k_new, X, y) - MSE_one_sample(theta_k, X, y)) < eps or np.linalg.norm(theta_k_new - theta_k) < eps):
                return (theta_k_new,k)
                break
                
            else:
                theta_k = theta_k_new
                if(k==k_max):
                    return "循环不收敛，计算失败."
```

## 4.小批量随机梯度下降
### (1) 基本思想
如果我们每次使用小批样本来计算损失函数，而不是一个样本或所有样本，则可以组合梯度下降和随机梯度下降的优势。

我们继续考虑MSE情况，每次循环我们都从$X$中随机抽取$m$个$x_i$（记为$x_{i_1},...,x_{i_m}$），并从$y$中找到其对应的$y_i$（记为$y_{i_1},...,y_{i_m}$）. 通常情况下，我们每“波”都要做$m$次迭代，并做$n_epoch$“波”.

令$X_b = (x_{i_1},...,x_{i_m})^T \in R^{m \times p}, y_b = (y_{i_1},...,y_{i_m})^T \in R^{m \times 1}$把它们输入到如下的公式中：

$$ \begin{aligned}MSE(\theta | X_b, y_b) = \frac{1}{m} \parallel X_b \theta - y_b\parallel^2= \frac{1}{m}\sum_{k=1}^{m}(\theta^T x_{i_k} - y_{i_k})^2 \end{aligned} $$

此时梯度为
{% raw %}
$$ \nabla MSE(\theta | X_b, y_b) = \frac{2}{m} X_b^T(X_b\theta-y_b) . $$
{% endraw %}

迭代公式为$\theta_{k+1} = \theta_k - step_k \nabla MSE(\theta | X_b, y_b)$. 注意：我们要把当前波数$epoch$也加入到学习计划中： $step_k = \frac{5}{epoch * n+k+50}$.
### (2) 代码


```python
def draw_m_samples_from_n(n, m):
    if(not isinstance(m, int) or m < 1 or m > n):
        return("Invalid batch size!")
    else:
        L = np.array([np.random.randint(1,n+1) for _ in range(m)])# This would produce m random integer from{1,2,...,n}
        return L
    
# 我们使用第一部分中提到的MSE函数和其梯度.
# def MSE(theta, X, y):
#     n = X.shape[0]
#     return 1/n * np.linalg.norm(X.dot(theta)-y)^2

# def MSE_gradient(theta, X, y):
#     n = X.shape[0]
#     return 2/n * X^T.dot(X.dot(theta)-y)

# 我们使用和第三部分相同的学习计划函数.
# def learning_schedule(t):
#    return 5/(t + 50)

def MSE_mini_batch_stochastic_gradient_descent(theta_0, X, y, eta, eps, n_epoch):
    theta_k = theta_0
    k = 0
    n = X.shape[0]
    
    for l in n_epoch:
        for k in range(m+1):
            L = draw_m_samples_from_n(n, m)
            g_k = MSE_gradient(theta_k, X[L], y[L])
        
            if(np.linalg.norm(g_k) < eps):
                return (theta_k,k)
                break
        
            else:
                eta = learning_schedule(l * n + k)
                theta_k_new = theta_k - eta * g_k
            
                if(np.linalg.norm(MSE(theta_k_new, X, y) - MSE(theta_k, X, y)) < eps or np.linalg.norm(theta_k_new - theta_k) < eps):
                    return (theta_k_new,k)
                    break
                
                else:
                    theta_k = theta_k_new
        
        if(l == n_epoch-1):
            return theta_k
```

## 参考资料：

1.统计学习方法（第2版）- 李航

2.机器学习 - 周志华

3.Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow, 2nd Edition - Aurélien Géron

4.Conjugate gradient method - Wikipedia


```python

```
