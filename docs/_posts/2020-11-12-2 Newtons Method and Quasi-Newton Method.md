---
author: stg1
catagory: blog
---

# *Introduction to Machine Learning* Algorithms and Realizations 2

## By Jiaheng Cui
>In this chapter, we'll focus on Newton's Method and DFP Quasi-Newton Method. If you are interested, you can also see how Zhihao Lyu used Newton's Method to solve the MLE of a Gamma distribution

<!-- more -->

## 1.Newton's Method
### (1) Basic thoughts
We introduced that gradient descent is a first-order optimization algorithm for finding a local minimum of a differentiable function in the last chapter. Now we introduce a second-order method, Newton's Method, or the Newton-Raphson Method.

We want to solve the following optimization problem:
$$ \mathop{\min}\limits_{x \in R^{n \times 1}} f(x)$$

Let $f(x)$ be **twice** differentiable, and $ x^{*} $ be the corresponding local minimum of $f(x)$.

The 2nd-order Taylor expansion of $f(x)$ is $f(x+\Delta x) = f(x) + g(x)\Delta x + \frac{1}{2} \Delta x^T H(x)\Delta x+ O(\parallel \Delta x \parallel^3)$, where $g(x) = \nabla f(x)$ is the gradient of $f$ at point $x$, $H(x) = \nabla^2 f(x)$ is the Hessian matrix of $f$ at point $x$. If we ignore the $O(\parallel \Delta x \parallel^3)$ term, then $f(x+\Delta x) \approx f(x) + \nabla f(x)\Delta x + \frac{1}{2} \Delta x^T H(x)\Delta x$, we can use this formula to derive an iterative method to solve $x^*$.

We'll choose a starting point $x_0$ and use the following equation to implement the iteration:
$x_{k+1} = x_k - H(x)^{-1} g_k = x_k - \left[ \nabla^2 f(x) \right]^{-1} \nabla f(x)$, $k = 0,1,2,...$

Again, if $f(x)$ is convex, then we have only one minimum point $x^{*}$, so the minimum we derived from gradient descent is $x^{*}$ itself (assuming there were no error).

### (2) Algorithm
Input: target function $f(x)$, gradient function $g(x) = \nabla f(x)$, Hessian matrix $H(x) = \nabla^2 f(x)$, tolerance $\epsilon$, starting point $x_0 \in R^{n \times 1}$, max iteration number $k_{max}$.

Output: a minimum point of $f(x)$, $x^{*}$.

Step1: load $x_0$, and set $k = 0$.

Step2: calculate $f(x_k)$

Step3: calculate $g_k = g(x_k)$, if $\parallel g_k\parallel \lt \epsilon$, stop iteration and let $x^* = x_k$; otherwise, let $p_k = -g_k$.

Step4: calculate $H_k = H(x_k)$, if $H_k$ is non-invertible, print "Hessian matrix non-invertible, calculation failed.", stop iteration; otherwise, calculate $H_{k}^{-1} = H^{-1}(x_k)$.

Step4: let $x_{k+1} = x_k + H_k^{-1} p_k$, calculate $f(x_{k+1})$.

Step5: if $\parallel f(x_{k+1}) - f(x_{k})\parallel \lt \epsilon$ or $\parallel x_{k+1} - x_{k}\parallel \lt \epsilon$, stop iteration and let $x^* = x_{k+1}$; otherwise, let $k = k+1$.

Step6: if $k = k_{max}$, the algorithm doesn't converge, print "Does not converge, calculation failed."; otherwise, return to Step3.

### (3) Code


```python
import numpy as np
```


```python
def Newton(f_x, g_x, H_x, x_0, eps = 1e-5, k_max = 1e5):
    x_k = x_0
    k = 0
    
    for k in range(k_max + 1):
        g_k = g_x(x_k)
        
        if(np.linalg.norm(g_k) < eps):
            return (x_k,k)
            break
        
        else:
            p_k = - g_k
            H_k = H_x(x_k)
            if(np.linalg.det(H_k) == 0):
                return "Hessian matrix non-invertible, calculation failed."
            
            else:
                x_k_new = x_k + np.linalg.inv(H_k).dot(p_k)
                if(np.linalg.norm(f_x(x_k_new) - f_x(x_k)) < eps or np.linalg.norm(x_k_new - x_k) < eps):
                    return (x_k_new,k)
                    break
                
                else:
                    x_k = x_k_new
                    if(k == k_max):
                        return "Does not converge, calculation failed."
```

Let $f(X) = (x_1 − x_2)^3 + (x_1 + 3x_2)^2$.

Then $g(x) = \nabla f(x) = (3(x_1 - x_2)^2 + 2(x_1 + 3x_2), -3(x_1 - x_2)^2 + 6(x_1 + 3x_2))^T$.

And $H(x) = \nabla^2 f(x) = \left[\begin{matrix} 6(x_1 - x_2) + 2 & -6(x_1 - x_2) + 6\\ -6(x_1 - x_2) + 6 & 6(x_1 - x_2) + 18\\ \end{matrix}\right]$.

Let the starting point be $x_0 = (1,2)^T$, we'll use Newton's method to find a minimum point of $f(x)$.


```python
x_0 = np.array([1,2])

def f_test(x):
    x_1 = x[0]
    x_2 = x[1]
    return ((x_1 - x_2) ** 3 + (x_1 + 3*x_2) ** 2)

def g_test(x):
    x_1 = x[0]
    x_2 = x[1]
    return np.array([3*(x_1 - x_2) ** 2 + 2*(x_1 + 3*x_2), -3*(x_1 - x_2) ** 2 + 6*(x_1 + 3*x_2)])

def H_test(x):
    x_1 = x[0]
    x_2 = x[1]
    return np.array([[6*(x_1 - x_2) + 2, -6*(x_1 - x_2) + 6], [-6*(x_1 - x_2) + 6, 6*(x_1 - x_2) + 18]])

print(Newton(f_test, g_test, H_test, x_0, 1e-5, 1000))
```

    (array([-0.00585938,  0.00195313]), 6)
    

### (4) Use Newton's Method to solve the MLE of gamma distribution

Please refer to the passage written by Zhihao Lyu:

https://github.com/jimcui3/Introduction-to-Machine-Learning/blob/main/2020-10-26-MLE%20with%20Newton%20Raphson.md

## 2.Quasi-Newton Method
### (1) Basic thoughts
Note that during the iteration, $H(x_k)$ is not always invertible. If there exists a $k$ s.t. $det(H(x_k)) = 0$, we cannot perform the Newton's method iteration. Thus there are many thoughts to use a series of invertible matrices to estimate **the inverse of the Hessian matrix**, we still denote these estimators as $\left\{ H_k \right\}$. These methods are called Quasi-Newton Methods. Everytime we'll use $H_k$ compute $x_{k+1}$, , then if didn't reach convergence, we'll get $H_{k+1}$ by a certain formula concerning $x_k$, $g_k$ and $H_k$. Then use $x_{k+1}$, $g_{k+1}$ and $H_{k+1}$ to compute $x_{k+2}$.

The most common Quasi-Newton Methods include DFP Quasi-Newton Method and BFGS Quasi-Newton Method. We'll only talk about DFP Quasi-Newton Method, those who are interested in BFGS Quasi-Newton Method can look it up here:

https://en.wikipedia.org/wiki/BFGS_method

### (2) DFP Quasi-Newton Method
Let $H_0 = I_n$, where $I_n$ is the identity matrix of order $n$.

Let $\Delta x_k = x_{k+1} - x_k$, $\Delta g_k = g_{k+1} - g_k$. Note that we do this step after we got $x_{k+1}$ (and thus $g_{k+1}$) from the iteration formula and only when we won't stop at $x_{k+1}$.

The iterative formula of $H_k$ is listed as follows:
$$H_{k+1} = H_k + \frac{\Delta x_k \Delta x_k^T}{\Delta x_k^T \Delta g_k} - \frac{H_k \Delta g_k \Delta g_k^T H_k}{\Delta g_k^T H_k \Delta g_k}$$

Note: It may be a little difficult to understand the double iteration loops and their sequential orders, if so, please review the following algorithm carefully and make sure you understand it clearly!

### (3) Algorithm
Input: target function $f(x)$, gradient function $g(x) = \nabla f(x)$, tolerance $\epsilon$, starting point $x_0 \in R^{n \times 1}$, max iteration number $k_{max}$.

Output: a minimum point of $f(x)$, $x^{*}$.

Step1: load $x_0$, and set $k = 0$, $H_0 = I$.

Step2: calculate $f(x_k)$

Step3: calculate $g_k = g(x_k)$, if $\parallel g_k\parallel \lt \epsilon$, stop iteration and let $x^* = x_k$; otherwise, continue.

Step4: let $x_{k+1} = x_k - H_k g_k$, calculate $f(x_{k+1})$.

Step5: if $\parallel f(x_{k+1}) - f(x_{k})\parallel \lt \epsilon$ or $\parallel x_{k+1} - x_{k}\parallel \lt \epsilon$, stop iteration and let $x^* = x_{k+1}$; otherwise, continue.

Step6: calculate $g_{k+1} = g(x_{k+1})$, if $\parallel g_{k+1}\parallel \lt \epsilon$, stop iteration and let $x^* = x_{l+1}; otherwise, $calculate $\Delta x_k = x_{k+1} - x_k$, $\Delta g_k = g_{k+1} - g_k$.

Step7: let $H_{k+1} = H_k + \frac{\Delta x_k \Delta x_k^T}{\Delta x_k^T \Delta g_k} - \frac{H_k \Delta g_k \Delta g_k^T H_k}{\Delta g_k^T H_k \Delta g_k}$, let $k = k+1$

Step8: if $k = k_{max}$, the algorithm doesn't converge, print "Does not converge, calculation failed."; otherwise, return to Step3.

### (4) Code


```python
def DFP_Quasi_Newton(f_x, g_x, x_0, eps = 1e-5, k_max = 1e5):
    x_k = x_0
    H_k = np.identity(x_0.shape[0])
    k = 0
    
    for k in range(k_max + 1):
        g_k = g_x(x_k)
        
        if(np.linalg.norm(g_k) < eps):
            return (x_k,k)
            break
        
        else:
            x_k_new = x_k - H_k.dot(g_k)
            
            if(np.linalg.norm(f_x(x_k_new) - f_x(x_k)) < eps or np.linalg.norm(x_k_new - x_k) < eps):
                return (x_k_new, k)
                break
                
            else:
                g_k_new = g_x(x_k_new)
        
                if(np.linalg.norm(g_k_new) < eps):
                    return (x_k_new, k)
                    break
                
                else:
                    delta_x_k = x_k_new - x_k
                    delta_g_k = g_k_new - g_k
                    H_k = H_k + np.outer(delta_x_k, delta_x_k) / delta_x_k.T.dot(delta_g_k) - H_k.dot(np.outer(delta_g_k, delta_g_k)).dot(H_k) / delta_g_k.T.dot(H_k).dot(delta_g_k)
                    x_k = x_k_new

                if(k == k_max):
                    return "Does not converge, calculation failed."
```

Let $f(X) = (4 − x_2)^3 + (x_1 + 4x_2)^2$.

Then $g(x) = \nabla f(x) = (2(x_1 + 4x_2), -3(4 - x_2)^2 + 8(x_1 + 4x_2))^T$.

Let the starting point be $x_0 = (2,1)^T$, we'll use Newton's method and DFP Quasi-Newton method to find a minimum point of $f(x)$.


```python
x_0 = np.array([2,1])

def f_test(x):
    x_1 = x[0]
    x_2 = x[1]
    return ((4 - x_2) ** 3 + (x_1 + 4*x_2) ** 2)

def g_test(x):
    x_1 = x[0]
    x_2 = x[1]
    return np.array([2*(x_1 + 4*x_2), -3*(4 - x_2) ** 2 + 8*(x_1 + 4*x_2)])

def H_test(x):
    x_1 = x[0]
    x_2 = x[1]
    return np.array([[2, 8], [8, 6*(4 - x_2) + 32]])

print(Newton(f_test, g_test, H_test, x_0, 1e-5, 1000))
print(DFP_Quasi_Newton(f_test, g_test, x_0, 1e-5, 1000))
```

    (array([-15.9765625 ,   3.99414062]), 8)
    (array([-15.9630806,   3.9907569]), 14)
    

As we can see, using DFP Quasi-Newton method is slower than regular Newton's method, however it's more stable.

Note: (-16,4) is only a local minimum ($gradient = 0$). $f(x)$ has no global minimum point, since when $x_1 = -4x_2$, and $x_2 \rightarrow + \infty$, $f(x) \rightarrow - \infty$.

## References:

1.统计学习方法（第2版）- 李航

2.最优化方法 - 杨庆之

3.Hessian Matrix - Wikipedia

4.Newton's method - Wikipedia


```python

```
