---
author: stg1
---
# 用牛顿迭代求解MLE
数理统计的作业，顺便也贴这，但是感觉锅很多...
## 问题
$X_1,..., X_n \sim Gamma(\alpha,\lambda)$，$f(x,\alpha,\lambda)=\frac{\lambda^\alpha}{\Gamma(\alpha)}x^{\alpha−1}e^{−\lambda x}$，根据样本估计参数$\alpha,\lambda$。

<!-- more -->

## 公式推导
### 极大似然
$$\begin{aligned}
L(\alpha,\lambda)&=(\frac{\lambda^\alpha}{\Gamma(\alpha)})^n\prod_{i=1}^{n}x_i^{\alpha−1}e^{−\lambda x_i}\\
l(\alpha,\lambda)&=n\alpha \ln \lambda - n\ln \Gamma(\alpha)+(\alpha-1)\sum_{i=1}^n\ln x_i-\lambda\sum_{i=1}^n x_i
\end{aligned}$$  
似然方程：  
注意到$\frac{\mathrm{d}}{\mathrm{d} x}\ln \Gamma(\alpha)$即为digamma函数$\psi(\alpha)$  
$$
\begin{cases}
\frac{\partial l}{\partial \alpha}= n\ln \lambda-n\psi(\alpha)+\sum_{i=1}^n\ln x_i=0\\
\frac{\partial l}{\partial \lambda}=\frac{n\alpha}{\lambda}-\sum_{i=1}^n x_i=0
\end{cases}
$$
### 矩估计
Gamma分布的数学期望$\mu=\frac{\alpha}{\lambda},\sigma^2=\frac{\alpha}{\lambda^2}$，故$\alpha,\lambda$矩估计为$\hat{\alpha}=\frac{\overline{\mu}^2}{S_n},\hat{\lambda}=\frac{\overline{\mu}}{S_n}$

### Newton-Raphson算法
$$
\begin{aligned}
\begin{bmatrix}
\hat{\alpha}^{(k+1)}\\
\hat{\lambda}^{(k+1)}
\end{bmatrix}
=\begin{bmatrix}
\hat{\alpha}^{(k)}\\
\hat{\lambda}^{(k)}
\end{bmatrix}
+H^{-1}
\begin{bmatrix}
\frac{\partial l}{\partial \alpha}(\hat{\alpha}^{(k)})\\
\frac{\partial l}{\partial \lambda}(\hat{\lambda}^{(k)})
\end{bmatrix}\\
H=-
\begin{bmatrix}
   -n\psi^{'}(\alpha) & \frac{n}{\lambda} \\
   \frac{n}{\lambda} & -\frac{n\alpha}{\lambda^2}
\end{bmatrix}
\end{aligned}
$$
### Fisher Scoring算法
Gamma分布是指数型分布族，$H(\hat{\theta}^{(k)})$与$x$无关，故Fisher Scoring和Newton-Raphson完全一致

## 数值计算
### 实现牛顿迭代


```python
import numpy as np
from scipy.special import polygamma, digamma
import matplotlib.pyplot as plt
import math

class Gamma:
    # 取alpha=5,lambda=2, 产生样本
    def __init__(self, alpha:float=10, lamb:float=0.1, n:float=1000):
        self.alpha = alpha
        self.lamb = lamb
        self.n = n
        self.data = np.random.gamma(alpha, 1/lamb, n)
        mean = np.mean(self.data)
        var = np.var(self.data, ddof=1)
        # 矩估计
        self.me_alpha = mean ** 2 / var
        self.me_lambda = mean / var
    
    def Newton(self, alpha0:float=None, lambda0:float=None, learning_rate:float=1.0,
               iteration:int=100, eps:float=1e-4, verbose:int=1):
        if (alpha0 == None):
            alpha0 = self.me_alpha
        if (lambda0 == None):
            lambda0 = self.me_lambda
        theta = np.array([alpha0, lambda0])
        tmp1 = np.sum(np.log(self.data))
        tmp2 = np.sum(self.data)
        for i in range(iteration):
            # Scipy中有计算digamma函数n重导数的polygamma函数，好像是用的Riemann-Zeta函数计算的
            H = np.array([
                [-self.n * polygamma(1, theta[0]), self.n/theta[1]],
                [self.n/theta[1], -self.n*theta[0]/(theta[1]**2)]
            ])
            s = np.array([
                self.n*math.log(theta[1]) - self.n*digamma(theta[0]) + tmp1,
                self.n*theta[0] / theta[1] - tmp2
            ])
            delta = np.linalg.inv(-H) @ s.T
            if (verbose == 2):
                print(np.linalg.inv(-H))
                print(s)
                print(delta)
                print(np.max(delta))
                print(np.max(delta) < eps)
            if (abs(np.max(delta)) < eps):
                if (verbose):
                    print(str(i)+'次迭代后收敛')
                iteration = i
                break
            theta += delta.T
        return {'alpha':theta[0], 'lambda':theta[1], 'iteration': iteration}
```

使用矩估计


```python
a = Gamma(n=1000)
print(a.me_alpha, a.me_lambda)
a.Newton(iteration=100,eps=1e-7)
```

    10.610932089452922 0.10720250369518106
    3次迭代后收敛
    {'alpha': 10.275468325113868, 'lambda': 0.1038133051654965, 'iteration': 3}



### 初值影响
上面的结果是以矩估计为真实值的，结果相对而言还算可以。  
当初值与真实值距离较远时，如果样本量n较大，则$s(\hat{\theta_k})$较大，则会出现牛顿迭代步长过长，出现直接迭代进负数导致无法继续计算的问题。  


```python
from random import uniform

def trail(a0_low,a0_up,l0_low,l0_up,times = 30):
    success = 0
    for i in range(times):
        a0=uniform(a0_low, a0_up)
        l0=uniform(l0_low, l0_up)
        try:
            ans = a.Newton(alpha0=a0, lambda0=l0, verbose=0)
            print('第'+str(i+1)+'次: alpha0',a0, 'lambda0', l0)
            print(ans)
            success+=1
        except:
            pass
    print('成功迭代率'+str(success/times*100)+'%')
```


```python
trail(1,20,0,0.5)
```

    第10次: alpha0 17.068797744939676 lambda0 0.1599353180536468
    {'alpha': 10.275452639304431, 'lambda': 0.10381315181133041, 'iteration': 6}
    第16次: alpha0 4.440171978099065 lambda0 0.05535445960033486
    {'alpha': 10.275458241660772, 'lambda': 0.10381320573978753, 'iteration': 5}
    第17次: alpha0 7.28032220854409 lambda0 0.06590695079542525
    {'alpha': 10.275459583345157, 'lambda': 0.10381320559678947, 'iteration': 4}
    成功迭代率10.0%


如上仅有一小部分迭代成功了。  
当随机初值的范围接近真实值时，迭代的成功率会增加很多


```python
trail(8,12,0.05,0.15)
```

    第3次: alpha0 11.599303499919667 lambda0 0.1060487972613175
    {'alpha': 10.275465254571172, 'lambda': 0.10381327602965268, 'iteration': 4}
    第6次: alpha0 10.830699880799303 lambda0 0.0889107153178688
    {'alpha': 10.275468323245645, 'lambda': 0.10381330514744759, 'iteration': 6}
    第8次: alpha0 10.547058627260338 lambda0 0.11584305308210266
    {'alpha': 10.275468196676572, 'lambda': 0.10381330368211261, 'iteration': 4}
    第12次: alpha0 8.201109267478053 lambda0 0.06962384965264577
    {'alpha': 10.275369505561745, 'lambda': 0.10381222849783105, 'iteration': 4}
    第13次: alpha0 8.319754558610413 lambda0 0.09196506056400244
    {'alpha': 10.275468270221932, 'lambda': 0.10381330462262213, 'iteration': 4}
    第14次: alpha0 10.333276378689852 lambda0 0.08884530352196254
    {'alpha': 10.275468320777438, 'lambda': 0.10381330512347312, 'iteration': 5}
    第17次: alpha0 9.955282798221301 lambda0 0.12094191095964728
    {'alpha': 10.275468315641291, 'lambda': 0.1038133050587898, 'iteration': 6}
    第20次: alpha0 8.522907896117575 lambda0 0.08948649335539036
    {'alpha': 10.275434353425256, 'lambda': 0.10381298493498012, 'iteration': 3}
    第21次: alpha0 9.16553236111829 lambda0 0.06931782993117377
    {'alpha': 10.275448520310556, 'lambda': 0.10381311192005127, 'iteration': 6}
    第22次: alpha0 10.248371630585714 lambda0 0.07928943531388757
    {'alpha': 10.275468321108303, 'lambda': 0.10381330512588843, 'iteration': 7}
    第30次: alpha0 11.673445046376155 lambda0 0.1359823920254255
    {'alpha': 10.275424223680384, 'lambda': 0.10381280469115983, 'iteration': 5}
    成功迭代率36.666666666666664%
