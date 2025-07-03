### ODE方程求解过程
目标向量场：dx/dt=v(x,t)


```python
import numpy as np #用于数值计算的库
from scipy.integrate import odeint #导入odeint函数
import matplotlib.pyplot as plt #用于绘图的库

# 定义微分方程
def model(y,t):
    dydt=-0.5*y #微分方程的定义，dydt表示y对时间t的导数
    return dydt

# 设置初始条件
y0 = 5

# 设置求解的时间范围
t=np.1 inspace(0,10,100) #从0到10产生100个时间点

solution=odeint(model,y0,t) #调用odeint.函数，计算微分方程的解

plt.plot(t,solution) #画出t变量与solution变量的关系
plt.xlabel('Time') #x轴标签
plt.ylabel('Value of y') #y轴标签
plt.title('ODE Solution with odeint') #图形标题
plt.grid() #添加网格
plt.show() #显示图形

```

```python

def compute_mu_t(self, t, x0, x1):
    """Compute the mean of time-dependent density p_t"""
    t = expand_t_like_x(t, x1) # 主要作用是将一维时间向量 t 扩展为多维张量，以便它可以与多维数据点 x 进行广播操作。
    alpha_t, _ = self.compute_alpha_t(t) # t, 1
    sigma_t, _ = self.compute_sigma_t(t) # 1 - t, -1
    if isinstance(x1, (list, tuple)):
        return [alpha_t[i] * x1[i] + sigma_t[i] * x0[i] for i in range(len(x1))]
    else:
        return alpha_t * x1 + sigma_t * x0  # xt=t*x1 + (1-t)*x0

def compute_xt(self, t, x0, x1):
    """Sample xt from time-dependent density p_t; rng is required"""
    xt = self.compute_mu_t(t, x0, x1)  # 如果需要随机采样，可以添加噪声项, SDE, return xt = xt + sigma_t * noise
    return xt

def compute_ut(self, t, x0, x1, xt):
    """Compute the vector field corresponding to p_t"""
    t = expand_t_like_x(t, x1)
    _, d_alpha_t = self.compute_alpha_t(t)
    _, d_sigma_t = self.compute_sigma_t(t)
    if isinstance(x1, (list, tuple)):
        return [d_alpha_t * x1[i] + d_sigma_t * x0[i] for i in range(len(x1))]
    else:
        return d_alpha_t * x1 + d_sigma_t * x0

def compute_ut2(self, t, x0, x1, xt):
    """d(phi)/dt=ut, u_t(x|x_1),计算目标向量场 xt=t*x1 + (1-t)*x0"""
    return (x1 - xt) / (1 - t)


def plan(self, t, x0, x1):
    xt = self.compute_xt(t, x0, x1)
    ut = self.compute_ut(t, x0, x1, xt) # dxt/dt = ut, 这里是不是有问题？
    return t, xt, ut

```

参考1:https://blog.csdn.net/shizheng_Li/article/details/147054031

参考2:https://blog.51cto.com/u_16213373/13142955
