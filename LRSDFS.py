import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from sympy.core.random import sample

import LRSDFS_func

# 初始化参数
k1 = 30 #低秩
k2 = 40 #稀疏
max_iter = 50  # 最大迭代次数
tol = 1e-6     # 收敛阈值

# 循环处理从 d00 到 d21 的数据
for i in range(22):  # 遍历 d00 到 d21
    train_file = 'TE_data/train_data/d00.dat'  # 格式化文件名
    #test_file = f'TE_data/test_data/d{}_te.dat'
    test_file = f'TE_data/test_data/d{i:02d}_te.dat'

    print(f"Processing Train Data: {train_file}, Test Data: {test_file}")

    # 加载训练和测试数据
    train_data = np.loadtxt(train_file)
    test_data = np.loadtxt(test_file)

    if(train_data.shape[0]<train_data.shape[1]):
        train_data=train_data.T
    if(test_data.shape[0]<test_data.shape[1]):
        test_data=test_data.T
    # 数据标准化
    train_data = train_data.T
    test_data = test_data.T

    n_samples, n_features = train_data.shape if train_data.shape[0] > train_data.shape[1] else train_data.shape[::-1]

    # 初始化字典矩阵 W1 和 W2
    W1 = np.random.normal(loc=50, scale=50, size=(n_samples, k1))
    W2 = np.random.normal(loc=50, scale=50, size=(n_samples, k2))

    # 初始化编码矩阵 Y1 和 Y2
    Y1 = np.random.normal(loc=50, scale=50, size=(k1, n_samples))
    Y2 = np.random.normal(loc=50, scale=50, size=(k2, n_samples))

    # 初始化用于 SVD 的矩阵 U1 和 U2
    U1 = np.random.uniform(low=-50, high=50, size=(k1, k1))
    U2 = np.random.uniform(low=-50, high=50, size=(k2, k2))

    # 初始化向量 i_d 和 i_c
    i_d = np.ones(n_features)
    i_c = np.ones(n_samples)

    # 记录每次迭代的范数
    W1_norms_e = []
    W2_norms_e = []
    Y1_norms_e = []
    Y2_norms_e = []

    prev_W1 = W1.copy()
    prev_W2 = W2.copy()
    prev_Y1 = Y1.copy()
    prev_Y2 = Y2.copy()

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
        Y1 = LRSDFS_func.update_Y1(X=train_data, W1=W1, W2=W2, Y1=Y1, Y2=Y2, U1=U1, c=0.05, e=8)

        # 更新 Y2
        Y2 = LRSDFS_func.update_Y2(X=train_data, W1=W1, W2=W2, Y1=Y1, Y2=Y2, U2=U2, i_d=i_d, i_c=i_c, b=0.5, d=0.01, f=8)

        # 更新 U1 和 U2
        U1, U2 = LRSDFS_func.update_U(Y1=Y1, Y2=Y2)

        # 保存每次的收敛误差
        W1_norms_e.append(np.linalg.norm(W1 - prev_W1))
        W2_norms_e.append(np.linalg.norm(W2 - prev_W2))
        Y1_norms_e.append(np.linalg.norm(Y1 - prev_Y1))
        Y2_norms_e.append(np.linalg.norm(Y2 - prev_Y2))

        # 检查收敛条件
        if np.linalg.norm(W1 - prev_W1) < tol and np.linalg.norm(W2 - prev_W2) < tol and \
                np.linalg.norm(Y1 - prev_Y1) < tol and np.linalg.norm(Y2 - prev_Y2) < tol:
            print("Convergence reached.")
            break

        # 更新上一次的字典矩阵
        prev_W1 = W1.copy()
        prev_W2 = W2.copy()
        prev_Y1 = Y1.copy()
        prev_Y2 = Y2.copy()

    # 计算字典矩阵
    D1 = train_data @ W1
    D2 = train_data @ W2
    #绘制字典热图
    plt.imshow(D1, cmap='hot')
    plt.colorbar()
    plt.title(f'Dictionary Matrix for {train_file}')
    plt.show()
    plt.imshow(D2, cmap='hot')
    plt.colorbar()
    plt.title(f'Dictionary Matrix for {train_file}')
    plt.show()

    #绘制收敛误差折线图
    plt.plot(W1_norms_e, label='W1')
    plt.plot(W2_norms_e, label='W2')
    plt.xlabel('Iterations')
    plt.ylabel('Convergence Error')
    plt.legend()
    plt.title(f'Convergence for {train_file}')
    plt.show()

    # 计算统计量
    T2_statistics, SPE_statistics = LRSDFS_func.calculate_statistics(X_new=test_data, D1=D1, D2=D2, a=0.5, b=0.5)

    # 使用核密度估计计算 T² 和 SPE 的控制限
    alpha = 0.99  # 置信水平
    #计算训练数据的统计量
    T2_train, SPE_train = LRSDFS_func.calculate_statistics(X_new=train_data, D1=D1, D2=D2, a=0.5, b=0.5)

    # T² 控制限
    kde_T2 = gaussian_kde(T2_train)
    T2_limit = np.percentile(T2_train, alpha * 100)

    # SPE 控制限
    kde_SPE = gaussian_kde(SPE_train)
    SPE_limit = np.percentile(SPE_train, alpha * 100)

    # 绘制 T² 统计量和控制限的折线图
    plt.plot(T2_statistics, label='T2')
    plt.axhline(y=T2_limit, color='r', linestyle='--', label='Control Limit')
    plt.xlabel('Sample Index')
    plt.ylabel('T2 Value')
    plt.legend()
    plt.title(f'T² Statistics for {test_file}')
    plt.show()

    # 绘制 SPE 统计量和控制限的折线图
    plt.plot(SPE_statistics, label='SPE')
    plt.axhline(y=SPE_limit, color='r', linestyle='--', label='Control Limit')
    plt.xlabel('Sample Index')
    plt.ylabel('SPE Value')
    plt.legend() # 显示图例
    plt.title(f'SPE Statistics for {test_file}')
    plt.show()