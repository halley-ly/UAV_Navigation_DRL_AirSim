# -*- coding: utf-8 -*-
"""
@对理想的一维匀加速直线运动模型，配有不精确的imu和不精确的gps，进行位置观测，协方差均使用矩阵的方式表示，以适配多维特征
"""
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(1,100,100) # 在1~100s内采样100次
u = 0.6 # 加速度值，匀加速直线运动模型
v0 = 5 # 初始速度
s0 = 0 # 初始位置
X_true = np.array([[s0], [v0]])
size = t.shape[0] + 1
dims = 2 # x, v, [位置, 速度]

Q = np.array([[1e1,0], [0,1e1]]) # 过程噪声的协方差矩阵，这是一个超参数
R = np.array([[1e4,0], [0,1e4]]) # 观测噪声的协方差矩阵，也是一个超参数。
# R_var = R.trace()
# 初始化
X = np.array([[0], [0]]) # 估计的初始状态，[位置, 速度]，就是我们要估计的内容，可以用v0，s0填入，也可以默认为0，相差越大，收敛时间越长
P = np.array([[0.1, 0], [0, 0.1]]) # 先验误差协方差矩阵的初始值，根据经验给出
# 已知的线性变换矩阵
F = np.array([[1, 1], [0, 1]]) # 状态转移矩阵
B = np.array([[1/2], [1]]) # 控制矩阵
H = np.array([[1,0],[0,1]]) # 观测矩阵

# 根据理想模型推导出来的真实位置值，实际生活中不会存在如此简单的运动模型，真实位置也不可知，本程序中使用真值的目的是模拟观测噪声数据和测量噪声数据
# 对于实际应用的卡尔曼滤波而言，并不需要知道真实值，而是通过预测值和观测值，来求解最优估计值，从而不断逼近该真值
real_positions = np.array([0] * size)
real_speeds = np.array([0] * size)
real_positions[0] = s0
# 实际观测值，通过理论值加上观测噪声模拟获得，初值即理论初始点加上观测噪声
measure_positions = np.array([0] * size)
measure_speeds = np.array([0] * size)
measure_positions[0] = real_positions[0] + np.random.normal(0, R[0][0]**0.5)
# 最优估计值，也就是卡尔曼滤波输出的真实值的近似逼近，同样地，初始值由观测值决定
optim_positions = np.array([0] * size)
optim_positions[0] = measure_positions[0]
optim_speeds = np.array([0] * size)

for i in range(1,t.shape[0]+1):
    # 根据理想模型获得当前的速度、位置真实值（实际应用中不需要），程序中只是为了模拟测试值和比较
    w = np.array([[np.random.normal(0, Q[0][0]**0.5)], [np.random.normal(0, Q[1][1]**0.5)]])
    X_true = F @ X_true + B * u + w
    real_positions[i] = X_true[0]
    real_speeds[i] = X_true[1]
    v = np.array([[np.random.normal(0, R[0][0]**0.5)], [np.random.normal(0, R[1][1]**0.5)]])
    # 观测矩阵用于产生真实的观测数据，注意各量之间的关联
    Z = H @ X_true + v
    # 以下是卡尔曼滤波的整个过程
    X_ = F @ X + B * u
    P_ = F @ P @ F.T + Q
    # 注意矩阵运算的顺序
    K = P_@ H.T @ np.linalg.inv(H @ P_@ H.T + R)
    X = X_ + K @ (Z - H @ X_)
    P = (np.eye(2) - K @ H ) @ P_
    # 记录结果
    optim_positions[i] = X[0][0]
    optim_speeds[i] = X[1][0]
    measure_positions[i] = Z[0]
    measure_speeds[i] = Z[1]
    
t = np.concatenate((np.array([0]), t))
plt.plot(t,real_positions,label='real positions')
plt.plot(t,measure_positions,label='measured positions')    
plt.plot(t,optim_positions,label='kalman filtered positions')

plt.legend()
plt.show()

plt.plot(t,real_speeds,label='real speeds')
plt.plot(t,measure_speeds,label='measured speeds')    
plt.plot(t,optim_speeds,label='kalman filtered speeds')

plt.legend()
plt.show()