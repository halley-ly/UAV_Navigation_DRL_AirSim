import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 配置中文字体支持和负号显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 创建 3D 图形
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('white')  # 背景设为白色

# 定义无人机当前位置 A（原点）和目标位置 B，以及 B 在水平面（z=0）的投影 C
A = np.array([0, 0, 0])        # 无人机当前位置
B = np.array([4, 2, 3])        # 目标位置（示例）
C = np.array([B[0], B[1], 0])  # B 在水平面上的投影

# 绘制 A、B、C 三个点，设置较大的 marker 以便观察
ax.scatter(A[0], A[1], A[2], color='navy', s=80, label='A（无人机）', depthshade=True)
ax.scatter(B[0], B[1], B[2], color='crimson', s=80, label='B（目标）', depthshade=True)
ax.scatter(C[0], C[1], C[2], color='darkgreen', s=80, label='C（水平投影）', depthshade=True)

# 绘制无人机前进方向（假设沿 x 轴正方向），使用箭头表示
ax.quiver(A[0], A[1], A[2], 3, 0, 0, color='black', arrow_length_ratio=0.1, linewidth=1.5, label='前进方向')

# 绘制 A 到 B 的直线（表示欧氏距离 D）
ax.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], 'k--', lw=2, label='D（欧氏距离）')

# 在 A 到 B 的中点处标注 D
D_mid = (A + B) / 2
ax.text(D_mid[0] + 0.1, D_mid[1] + 0.3, D_mid[2] + 0.2
        , 'D', color='black', fontsize=14)

# 绘制 B 到 C 的垂直线（表示垂直距离 Z）
ax.plot([B[0], C[0]], [B[1], C[1]], [B[2], C[2]], 'g--', lw=2, label='Z（垂直距离）')

# 在 B 到 C 的中点处标注 Z
Z_mid = (B + C) / 2
ax.text(Z_mid[0] + 0.1, Z_mid[1] + 0.1, Z_mid[2] + 0.1, 'Z', color='green', fontsize=14)

# 绘制 A 到 C 的水平连线（用于计算角度 β）
ax.plot([A[0], C[0]], [A[1], C[1]], [A[2], C[2]], 'b--', lw=2)

# 计算 β：无人机前进方向（x 轴正方向）与 A 到 C 水平连线之间的夹角
v_horizontal = np.array([B[0], B[1]])  # A->B 在水平面的投影
beta = np.arctan2(v_horizontal[1], v_horizontal[0])  # 单位：弧度
beta_deg = np.degrees(beta)

# 在水平面上（z=0）绘制 β 弧线
arc_radius = 0.8
theta = np.linspace(0, beta, 100)
arc_x = arc_radius * np.cos(theta)
arc_y = arc_radius * np.sin(theta)
arc_z = np.zeros_like(arc_x)
ax.plot(arc_x, arc_y, arc_z, color='magenta', lw=2, label='β（夹角）')
# 在弧线中间标注 β
ax.text(arc_radius * np.cos(beta/2) + 0.1, 
        arc_radius * np.sin(beta/2) + 0.1, 
        0, r'$\beta$', color='magenta', fontsize=14)

# 设置坐标轴标签和标题
ax.set_xlabel('X', fontsize=12, labelpad=10)
ax.set_ylabel('Y', fontsize=12, labelpad=10)
ax.set_zlabel('Z', fontsize=12, labelpad=10)
ax.set_title('无人机状态信息三维示意图\n$S_{state} = [D, Z, \\beta]$', fontsize=16, pad=20)

# 去除坐标轴上的刻度数字
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

# 设置坐标轴范围
ax.set_xlim(-1, 6)
ax.set_ylim(-1, 6)
ax.set_zlim(-1, 6)

# 显示网格，并调整图例样式
ax.grid(True)
legend = ax.legend(loc='upper left', bbox_to_anchor=(-0.2, 1.05), fontsize=12, frameon=True)
legend.get_frame().set_alpha(0.8)

plt.tight_layout()
plt.show()
