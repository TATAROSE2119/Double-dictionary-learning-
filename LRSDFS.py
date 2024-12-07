import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.bipartite.basic import color

import LRSDFS_func

# Load train data
train_data = np.loadtxt('TE_data/train_data/d02.dat')
test_data = np.loadtxt('TE_data/test_data/d02_te.dat')

# 数据标准化
train_data = (train_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0)
test_data = (test_data - np.mean(test_data, axis=0)) / np.std(test_data, axis=0)

# 打印数据维度
print(train_data.shape)

# 初始化字典矩阵 W1 和 W2
k1 = 52
k2 = 52
W1 = np.random.rand(52, k1)
W2 = np.random.rand(52, k2)

# 初始化编码矩阵 Y1 和 Y2
Y1 = np.random.rand(k1, 52)
Y2 = np.random.rand(k2, 52)

# 初始化用于 SVD 的矩阵 U1 和 U2
U1 = np.random.rand(k1, k1)
U2 = np.random.rand(k2, k2)

# 初始化向量 i_d 和 i_c
i_d = np.ones(480)
i_c = np.ones(52)

# 设置迭代参数
max_iter = 1 # 最大迭代次数
tol = 1e-6      # 收敛阈值
prev_W1 = W1.copy()
prev_W2 = W2.copy()

# 记录每次迭代的范数
W1_norms = []
W2_norms = []
Y1_norms = []
Y2_norms = []

# 迭代更新
for iteration in range(max_iter):
    print(f"Iteration: {iteration + 1}")

    # 对 train_data @ W1 进行 SVD
    XW_1 = train_data @ W1
    M, S1, N = np.linalg.svd(XW_1, full_matrices=False)
    N = N.T

    # 更新 W1
    W1 = LRSDFS_func.update_W1(X=train_data, W1=W1, W2=W2, Y1=Y1, Y2=Y2, M=M, N=N, a=0.5)

    # 更新 W2
    W2 = LRSDFS_func.update_W2(X=train_data, W1=W1, W2=W2, Y1=Y1, Y2=Y2, i_d=i_d, i_c=i_c, b=0.5)

    # 更新 Y1
    Y1 = LRSDFS_func.update_Y1(X=train_data, W1=W1, W2=W2, Y1=Y1, Y2=Y2, U1=U1, c=0.5, e=0.1)

    # 更新 Y2
    Y2 = LRSDFS_func.update_Y2(X=train_data, W1=W1, W2=W2, Y1=Y1, Y2=Y2, U2=U2, i_d=i_d, i_c=i_c, b=0.5, d=0.1, f=0.2)

    # 更新 U1 和 U2
    U1, U2 = LRSDFS_func.update_U(Y1=Y1, Y2=Y2)

    # 记录当前范数
    W1_norms.append(np.linalg.norm(W1))
    W2_norms.append(np.linalg.norm(W2))
    Y1_norms.append(np.linalg.norm(Y1))
    Y2_norms.append(np.linalg.norm(Y2))

    # 检查收敛条件
    if np.linalg.norm(W1 - prev_W1) < tol and np.linalg.norm(W2 - prev_W2) < tol:
        print("Convergence reached.")
        break
    # 没有收敛
    print("Not Convergence.")
    # 更新上一次的字典矩阵
    prev_W1 = W1.copy()
    prev_W2 = W2.copy()

# 绘制范数收敛折线图
plt.figure(figsize=(10, 6))
plt.plot(range(len(W1_norms)), W1_norms, label="W1 Norm")
plt.plot(range(len(W2_norms)), W2_norms, label="W2 Norm")
plt.plot(range(len(Y1_norms)), Y1_norms, label="Y1 Norm")
plt.plot(range(len(Y2_norms)), Y2_norms, label="Y2 Norm")
plt.xlabel("Iteration")
plt.ylabel("Norm")
plt.title("Convergence of Matrix Norms")
plt.legend()
plt.grid(True)
#保存图片
plt.savefig('LRSDFS_convergence.png')

# 假设 l = 10
# l = 10
# # 计算特征重要性
# p = 0.5  # 权重参数
# q = 0.5
# feature_importance = LRSDFS_func.calculate_feature_importance(Y1=Y1, Y2=Y2, p=p, q=q)
# # 打印特征重要性
# print("Feature Importance:")
# print(feature_importance)
# # 选择前 l 个特征
# Xnew, selected_features_indices = LRSDFS_func.select_top_features(train_data, feature_importance, l)
#
# # 打印结果
# print("Selected Feature Indices (Top l):", selected_features_indices)
# print("Shape of Xnew:", Xnew.shape)


# 示例：加载新的 960 个样本的数据
new_data = test_data

# 计算协方差矩阵（在训练阶段完成一次即可）
Sigma_Y1 = np.cov(Y1.T)  # Y1 的协方差矩阵
Sigma_Y2 = np.cov(Y2.T)  # Y2 的协方差矩阵

# 计算 T² 和 SPE 统计量
T2_statistics, SPE_statistics = LRSDFS_func.calculate_statistics(train_data,new_data, W1, W2, Y1, Y2, Sigma_Y1, Sigma_Y2)
#使用KDE计算控制限
T2_control_limit = LRSDFS_func.calculate_control_limit(T2_statistics, percentile=99)
SPE_control_limit = LRSDFS_func.calculate_control_limit(SPE_statistics,percentile=99)

# 统计图
# 绘制 T² 统计量折线图
plt.figure(figsize=(10, 5))
plt.plot(range(len(T2_statistics)), T2_statistics, label="T² Statistics",color='blue')
plt.axhline(y=T2_control_limit, color='r', linestyle='--', label=f"Control Limit ({T2_control_limit:.4f})")
plt.xlabel("Sample Index")
plt.ylabel("T² Value")
plt.title("T² Statistics Line Chart")
plt.legend()
plt.grid(True)
# 保存图片
plt.savefig('LRSDFS_T2.png')

# 绘制 SPE 统计量折线图
plt.figure(figsize=(10, 5))
plt.plot(range(len(SPE_statistics)), SPE_statistics, label="SPE Statistics", color='blue')
plt.axhline(y=SPE_control_limit, color='r', linestyle='--', label=f"Control Limit ({SPE_control_limit:.4f})")
plt.xlabel("Sample Index")
plt.ylabel("SPE Value")
plt.title("SPE Statistics Line Chart")
plt.legend()
plt.grid(True)
# 保存图片
plt.savefig('LRSDFS_SPE.png')