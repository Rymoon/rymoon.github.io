---
author: stg1
catagory: blog
---

# *Introduction to Machine Learning* Algorithms and Realizations 1

## By Jiaheng Cui
>In this chapter, we'll focus on an optimization method - gradient descent, and we'll explain how gradient descent works in fine-tuning models. Finally we'll introduce some revised method of gradient descent.

<!-- more -->

## 1.Gradient Descent
### (1) Basic thoughts
Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function.

We want to solve the following optimization problem:
$$ \mathop{\min}\limits_{x \in R^n} f(x)$$

Let $f(x)$ be differentiable, and $ x^{*} $ be the corresponding local minimum of $f(x)$.

We'll choose a starting point $x_0$ and use the following equation to implement the iteration:
$$x_{k+1} = x_k + step_k p_k, k = 0,1,2,...$$

Where $step_k$ is the k-th step size (or learning rate), and $p_k$ is the iteration direction.

In Gradient Descent, as $f(x)$ **decreases the fastest** if one goes from $x_k$ in the direction of the **negative** gradient of $f$ at $x_k$, which is $-\nabla f(x_k)$, we'll let $p_k = -\nabla f(x_k)$ at the k-th iteration.

Then we can use line search to solve $step_k$, that is, let $step_k = \mathop{\arg\min}\limits_{step_k \ge 0} f(x_k + step_k p_k)$. Or we can use grid search to find a fixed good step size.

Note that if $f(x)$ is convex, then we have only one minimum point $x^{*}$, so the minimum we derived from gradient descent is $x^{*}$ itself (assuming there were no error).

### (2) Quadratic situation
If $f(x)$ is positive definite and quadratic, that is $$f(x) = \frac{1}{2}x^T A x + b^T x + c, A \gt 0$$

Where $A \gt 0$ means $A$ is positive definite, and $x$ here is a vector. Then $f(x)$ is a convex function on $R^{n \times 1}$.

Let $g_k = \nabla f(x_k) = Ax_k + b$, then the iteration direction $p_k = -g_k$, now $x_{k+1} = x_k - step_k g_k$.

Then we can derive an explicit formula of $step_k$. $step_k$ would make $f(x)$ to its minimum at direction $g_k$, so $step_k$ should be a minimizer of $\phi(\alpha) = f(x_k - \alpha g_k)$.

Since $\phi'(\alpha) = ((x_k - \alpha g_k)^T A + b^T) (-g_k) $ and $\phi'(step_k) = 0$, $step_k = \frac{g_k^T g_k}{g_k^T A g_k}$. 

As a result, $$x_{k+1} = x_k - \frac{g_k^T g_k}{g_k^T A g_k} g_k, g_k = Ax_k + b$$

### (3) Algorithm
Input: target function $f(x)$, gradient function $g(x) = \nabla f(x)$, tolerance $\epsilon$, starting point $x_0 \in R^{n \times 1}$, max iteration number $k_{max}$.

Output: a minimum point of $f(x)$, $x^{*}$.

Step1: load $x_0$, and set $k = 0$.

Step2: calculate $f(x_k)$

Step3: calculate $g_k = g(x_k)$, if $\parallel g_k\parallel \lt \epsilon$, stop iteration and let $x^* = x_k$; otherwise, let $p_k = -g_k$ and solve $step_k = \mathop{\arg\min}\limits_{step_k \ge 0} f(x_k + step_k p_k)$.

**Note: in this chapter, $\parallel \parallel$ means $L^2$ norm.**

Step4: let $x_{k+1} = x_k + step_k p_k$, calculate $f(x_{k+1})$.

Step5: if $\parallel f(x_{k+1}) - f(x_{k})\parallel \lt \epsilon$ or $\parallel x_{k+1} - x_{k}\parallel \lt \epsilon$, stop iteration and let $x^* = x_{k+1}$; otherwise, let $k = k+1$.

Step6: if $k = k_{max}$, the algorithm doesn't converge, print "Does not converge, calculation failed."; otherwise, return to Step3.

### (4) Code


```python
import numpy as np
```

Generally, solving $step_k$ would be inconvenient. We'll just show two situations: fixed learning rate and quadratic situation.

<1> Fixed learning rate $\eta$ given by user, then $step_k = \eta, k=0,1,2,...$:


```python
def gradient_descent_fixed_eta(f_x, g_x, eps, x_0, eta, k_max):
    x_k = x_0
    k = 0
    
    for k in range(k_max + 1):#range(n) would produce a sequence: 0, 1, ..., n-1
        g_k = g_x(x_k)
        
        if(np.linalg.norm(g_k) < eps):#np.linalg.norm(g_k) is the L2 norm of g_k
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
                if(k == k_max):
                    return "Does not converge, calculation failed."
```

<2> Quadratic situation, $A$ should be given by user:


```python
def gradient_descent_quadratic(f_x, g_x, eps, x_0, A, k_max):
    x_k = x_0
    k = 0
    
    for k in range(k_max + 1):
        g_k = g_x(x_k)
        
        if(np.linalg.norm(g_k) < eps):
            return (x_k,k)
            break
            
        else:
            step_k = g_k.T.dot(g_k)/g_k.T.dot(A).dot(g_k)#.T is the action of matrix transpose, .dot() is the action of matrix multiplication
            x_k_new = x_k - step_k * g_k
            
            if(np.linalg.norm(f_x(x_k_new) - f_x(x_k)) < eps or np.linalg.norm(x_k_new - x_k) < eps):
                return (x_k_new,k)
                break
                
            else:
                x_k = x_k_new
                if(k == k_max):
                    return "Does not converge, calculation failed."
```

Let's do a quick test. Let $f(x_1, x_2) = x_1^2 + 4x_2^2$. Obviously $f(x)$ is convex, and the only minimum is $(0,0)$.

We can convert $f(x_1, x_2)$ into its matrix form: $f(x) = \frac{1}{2}x^T Ax, where x = (x_1,x_2)^T, A = \left[\begin{matrix} 2 & 0\\ 0 & 8\\ \end{matrix}\right] $. Then $g(x) = \nabla f(x) = Ax$.


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
    

As we can see, if $f(x)$ is a quadratic function, it's better to use the explicit $step_k$. The iteration number using $step_k$ is 6 while using fixed learning rate is 24.

### (5) What can we do with Gradient Descent in Machine Learning?

In machine learning tasks, we usually have a training set $S = \left\{(x_1,y_1),(x_2,y_2),...,(x_n,y_n)\right\}, where x_i \in R^{p \times 1} ,y_i \in R$, a model with parameter vector $\theta = (\theta_1, \theta_2, ... ,\theta_p)^T \in R^{p \times 1}$. And we want to minimize the loss function $L(\theta) = \frac{1}{n}\sum_{i=1}^{n}L(f(x_i,\theta),y_i)$.

Suppose we want to train a linear model and choose MSE as our loss function:
$$MSE(\theta) = \frac{1}{n} \parallel X \theta - y\parallel^2= \frac{1}{n}\sum_{i=1}^{n}(\theta^T x_i - y_i)^2$$

Where$X = (x_1, x_2, ..., x_n)^T \in R^{n \times p}, y = (y_1, y_2, ..., y_n)^T \in R^{n \times 1}$.

If we want to minimize MSE by gradient descent, we need to compute its gradient:
$$\nabla MSE(\theta) = \frac{2}{n} X^T(X\theta-y)$$

The iteration formula is $\theta_{k+1} = \theta_k - step_k \nabla MSE(\theta)$.

If we put MSE and its gradient into python code, then we can use gradient descent to find the best $\theta$ of our model. We'll use a fixed step size $\eta$ given by the user:


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
                    return "Does not converge, calculation failed."
```

## 2.Conjugate Gradient Descent
### (1) Quadratic situation
When $x_k$ is near $x^*$, especially when $f(x)$ is complicated, convergence would be slower and slower. Conjugate gradient descent can perform better when this problem occurs.

Let $A$ be a positive definite matrix, $Q_1$ and $Q_2$ are two non-zero vector. Then $Q_1$ and $Q_2$ are conjugate with respect to $A$ if $Q_1^T A Q_2 = 0$.

We'll still consider the quadratic problem:$$ \mathop{\min}\limits_{x \in R^n} f(x) = \mathop{\min}\limits_{x \in R^n}\frac{1}{2}x^T A x + b^T x + c, A \gt 0$$

The iteration formula here is $x_{k+1} = x_k + \alpha_k p_k$.

Previously, $p_{k+1} = -\nabla f(x_{k+1})$. Now we can use $step_k$ and $p_k$ to revise $p_{k+1}$:
$$p_{k+1} =  -\nabla f(x_{k+1}) + step_k p_k, p_0 = -\nabla f(x_0)$$

Let $p_{k+1}$ and $p_k$ be conjugate, then $0 = p_{k+1}^T A p_k = -\nabla f(x_{k+1})A p_k + step_k p_k^TA p_k$. Then we can get $step_k = \frac{-\nabla f(x_{k+1})^T A p_k}{p_k^TA p_k}$.

Then by line search can we get $\alpha_k = \frac{p_k^T(-b-Ax_k)}{p_k^T A p_k}$.

As a result, in every iteration we need to first update $p_k = - \nabla f(x_k) - \frac{\nabla f(x_{k})^T A p_{k-1}}{p_{k-1}^TA p_{k-1}}p_{k-1}$. Then use the new $p_k$ to calculate $\alpha_k = \frac{p_k^T(-b-Ax_k)}{p_k^T A p_k}$. Finally get $x_{k+1} = x_k + \alpha_k p_k$.

### (2) Algorithm
Input: target function $f(x)$, gradient function $g(x) = \nabla f(x)$, tolerance $\epsilon$, starting point $x_0 \in R^{n \times 1}$, positive definite matrix $A \in R^{n \times n}$ used in $f(x)$, vector $b \in R^{n \times 1}$ used in $f(x)$, max iteration number $k_{max}$.

Output: a minimum point of $f(x)$, $x^{*}$.

Step1: load $x_0$, and set $k = 0$.

Step2: calculate $f(x_k)$.

Step3: calculate $g_k = g(x_k)$, if $\parallel g_k\parallel \lt \epsilon$, stop iteration and let $x^* = x_k$; otherwise, if $k = 0$, let $p_k = -g_k$, else if $k > 0$, let $p_k = -g_k  - \frac{g_k^T A p_{k-1}}{p_{k-1}^TA p_{k-1}}p_{k-1}$ and solve $\alpha_k = \frac{p_k^T(-b-Ax_k)}{p_k^T A p_k}$.

Step4: let $x_{k+1} = x_k + \alpha_k p_k$, calculate $f(x_{k+1})$.

Step5: if $\parallel f(x_{k+1}) - f(x_{k})\parallel \lt \epsilon$ or $\parallel x_{k+1} - x_{k}\parallel \lt \epsilon$, stop iteration and let $x^* = x_{k+1}$; otherwise, let $k = k+1$.

Step6: if $k = k_{max}$, the algorithm doesn't converge, print "Does not converge, calculation failed.";otherwise, return to Step3.

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
                return "Does not converge, calculation failed."
```

Let's do a quick test. Let $f(x_1, x_2) = x_1^2 - 2x_1 + 4x_2^2 - 16x_2$. Obviously $f(x)$ is convex, and the only minimum is $(1,2)$.

We can convert $f(x_1, x_2)$ into its matrix form: $f(x) = \frac{1}{2}x^T Ax + b^Tx, where x = (x_1,x_2)^T, A = \left[\begin{matrix} 2 & 0\\ 0 & 8\\ \end{matrix}\right], b = (-2, -16)^T $. Then $g(x) = \nabla f(x) = Ax + b$.


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
    

## 3.Stochastic Gradient Descent
### (1) Basic thoughts
In part 1, we can see when computing loss function, we need to use $x_1, x_2, ..., x_n$, i.e. all samples. Hence when $n$ is large, it'll be very slow to compute loss function and its gradient.

We now introduce Stochastic Gradient Descent, which only use one random sample to compute loss function and its gradient. This would make the algorithm much faster when dealing with huge training sets, and possibly to jump out of the neighborhood of a local minimum and find the global minimum. However, when it end up close to the minimum, it'll bounce around, thus never find the optimal parameter but only a relatively good parameter.

We'll deal with the MSE situation. Every iteration we'll randomly pick out one $x_i$ from $X$, and corresponding $y_i$ from $y$, then input them into the following formula:

$$MSE(\theta | x_i, y_i) = (\theta^T x_i - y_i)^2$$

Then we can compute the gradient: $\nabla MSE(\theta | x_i, y_i) = 2(x_i^T\theta-y_i) x_i$.

The iteration formula is $\theta_{k+1} = \theta_k - step_k \nabla MSE(\theta | x_i, y_i)$. Here we'll use a simple learning schedule: $step_k = \frac{5}{n+k+50}$, this would make step size smaller and smaller and restrain $\theta$ from bouncing around.
### (2) Code


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
        i = np.random.randint(1, n+1)#randint(1, n+1) will produce a random integer from {1,2,...,n}
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
                    return "Does not converge, calculation failed."
```

## 4.Mini-batch Stochastic Gradient Descent
### (1) Basic thoughts
If we use small batches of samples every time to calculate loss function rather than one sample or all samples, we could somehow combine the advantages of gradient descent and stochastic gradient descent.

We'll still deal with the MSE situation. Every iteration we'll randomly pick out $m (1\le m \le n)$ random $x_i$s (denoted as $x_{i_1},...,x_{i_m}$) from $X$, and corresponding $y_i$s (denoted as $y_{i_1},...,y_{i_m}$) from $y$. By convention we iterate by rounds of $m$ iterations, each round is called an epoch, we'll do $n_epoch$ epochs.

Let $X_b = (x_{i_1},...,x_{i_m})^T \in R^{m \times p}, y_b = (y_{i_1},...,y_{i_m})^T \in R^{m \times 1}$, then input them into the following formula:

$$MSE(\theta | X_b, y_b) = \frac{1}{m} \parallel X_b \theta - y_b\parallel^2= \frac{1}{m}\sum_{k=1}^{m}(\theta^T x_{i_k} - y_{i_k})^2$$

Then we can compute the gradient: $$\nabla MSE(\theta | X_b, y_b) = \frac{2}{m} X_b^T(X_b\theta-y_b)$$.

The iteration formula is $\theta_{k+1} = \theta_k - step_k \nabla MSE(\theta | X_b, y_b)$. Note that we'll plug the current epoch number into the learning schedule: $step_k = \frac{5}{epoch*n+k+50}$.
### (2) Code


```python
def draw_m_samples_from_n(n, m):
    if(not isinstance(m, int) or m < 1 or m > n):
        return("Invalid batch size!")
    else:
        L = np.array([np.random.randint(1,n+1) for _ in range(m)])# This would produce m random integer from{1,2,...,n}
        return L
    
# Note that we'll use the same MSE and gradient as part 1.
# def MSE(theta, X, y):
#     n = X.shape[0]
#     return 1/n * np.linalg.norm(X.dot(theta)-y)^2

# def MSE_gradient(theta, X, y):
#     n = X.shape[0]
#     return 2/n * X^T.dot(X.dot(theta)-y)

# We'll use the same learning schedule as part 3.
# def learning_schedule(t):
#    return 5/(t + 50)

def MSE_mini_batch_stochastic_gradient_descent(theta_0, X, y, eta, eps, k_max, n_epoch):
    theta_k = theta_0
    k = 0
    n = X.shape[0]
    
    for n in n_epoch:
        for k in range(k_max+1):
            L = draw_m_samples_from_n(n, m)
            g_k = MSE_gradient(theta_k, X[L], y[L])
        
            if(np.linalg.norm(g_k) < eps):
                return (theta_k,k)
                break
        
            else:
                eta = learning_schedule(n + k)
                theta_k_new = theta_k - eta * g_k
            
                if(np.linalg.norm(MSE(theta_k_new, X, y) - MSE(theta_k, X, y)) < eps or np.linalg.norm(theta_k_new - theta_k) < eps):
                    return (theta_k_new,k)
                    break
                
                else:
                    theta_k = theta_k_new
                    if(k==k_max):
                        return "Does not converge, calculation failed."
```

## References:

1.统计学习方法（第2版）- 李航

2.机器学习 - 周志华

3.Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow, 2nd Edition - Aurélien Géron

4.Conjugate gradient method - Wikipedia
