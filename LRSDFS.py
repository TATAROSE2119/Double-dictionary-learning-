import numpy as np
import LRSDFS_func

# Load train data
train_data = np.loadtxt('TE_data/train_data/d01.dat')
test_data = np.loadtxt('TE_data/test_data/d01_te.dat')

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
max_iter = 10  # 最大迭代次数
tol = 1e-6      # 收敛阈值
prev_W1 = W1.copy()
prev_W2 = W2.copy()

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

    # 检查收敛条件
    if np.linalg.norm(W1 - prev_W1) < tol and np.linalg.norm(W2 - prev_W2) < tol:
        print("Convergence reached.")
        break
    #没有收敛
    print("Not Convergence.")
    # 更新上一次的字典矩阵
    prev_W1 = W1.copy()
    prev_W2 = W2.copy()


