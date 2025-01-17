import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from statsmodels.graphics.tukeyplot import results
from tqdm import tqdm
from sklearn.neighbors import KernelDensity
from sympy.core.random import sample

import LRSDFS_func

if not os.path.exists('plts/convergence_error'):
    os.makedirs('plts/convergence_error')
# 如果保存路径不存在则新建路径
if not os.path.exists('plts/statistics'):
    os.makedirs('plts/statistics')
#如果保存路径不存在则新建路径
if not os.path.exists('plts/dictionary'):
    os.makedirs('plts/dictionary')
if not os.path.exists('convergence_error'):
    os.makedirs('convergence_error')
# 初始化参数
k1 = 52 #低秩
k2 = 52 #稀疏
max_iter = 500  # 最大迭代次数
tol = 1e-3     # 收敛阈值
train_file = 'TE_data/train_data/d00.dat'  # 格式化文件名
train_data = np.loadtxt(train_file)
if(train_data.shape[0]<train_data.shape[1]):
        train_data=train_data.T
train_data = train_data.T
n_samples, n_features = train_data.shape if train_data.shape[0] > train_data.shape[1] else train_data.shape[::-1]


results=[]
# 循环处理从 d00 到 d21 的数据
for i in range(1,22):  # 遍历 d00 到 d21

    #test_file = f'TE_data/test_data/d{}_te.dat'
    test_file = f'TE_data/test_data/d{i:02d}_te.dat'

    print(f"Processing Train Data: {train_file}, Test Data: {test_file}")

    # 加载训练和测试数据

    test_data = np.loadtxt(test_file)


    if(test_data.shape[0]<test_data.shape[1]):
        test_data=test_data.T
    # 数据标准化

    test_data = test_data.T



    # 初始化字典矩阵 W1 和 W2
    W1 = np.random.rand(n_samples, k1)
    W2 = np.random.rand(n_samples, k2)

    # 初始化编码矩阵 Y1 和 Y2
    Y1 = np.random.rand(k1, n_samples)
    Y2 = np.random.rand(k2, n_samples)

    # 初始化用于 SVD 的矩阵 U1 和 U2
    U1 = np.random.rand(k1, k1)
    U2 = np.random.rand(k2, k2)

    # 初始化向量 i_d 和 i_c
    i_d = np.ones(n_features)
    i_c = np.ones(n_samples)

    # 记录每次迭代的范数
    W1_norms_e = []
    W2_norms_e = []
    Y1_norms_e = []
    Y2_norms_e = []


    # 迭代更新
    for iteration in tqdm(range(max_iter),desc=f"Processing file {train_file}"):
        #print(f"Iteration: {iteration + 1}")

        # 对 train_data @ W1 进行 SVD
        XW_1 = train_data @ W1
        M, S1, N = np.linalg.svd(XW_1, full_matrices=False)
        N = N.T

        prev_W1 = W1.copy()
        prev_W2 = W2.copy()
        prev_Y1 = Y1.copy()
        prev_Y2 = Y2.copy()

        # 更新 W1
        W1 = LRSDFS_func.update_W1(X=train_data, W1=W1, W2=W2, Y1=Y1, Y2=Y2, M=M, N=N, a=0.3)

        # 更新 W2
        W2 = LRSDFS_func.update_W2(X=train_data, W1=W1, W2=W2, Y1=Y1, Y2=Y2, i_d=i_d, i_c=i_c, b=0.1)

        # 更新 Y1
        Y1 = LRSDFS_func.update_Y1(X=train_data, W1=W1, W2=W2, Y1=Y1, Y2=Y2, U1=U1, c=1e5, e=10)

        # 更新 Y2
        Y2 = LRSDFS_func.update_Y2(X=train_data, W1=W1, W2=W2, Y1=Y1, Y2=Y2, U2=U2, i_d=i_d, i_c=i_c, b=0.3, d=1e5, f=10)

        # 更新 U1 和 U2
        U1, U2 = LRSDFS_func.update_U(Y1=Y1, Y2=Y2)


        # 保存每次的收敛误差
        W1_norms_e.append(np.linalg.norm(W1 - prev_W1))
        W2_norms_e.append(np.linalg.norm(W2 - prev_W2))
        Y1_norms_e.append(np.linalg.norm(Y1 - prev_Y1))
        Y2_norms_e.append(np.linalg.norm(Y2 - prev_Y2))

        # 检查收敛条件
        if LRSDFS_func.calculate_relerr(W1, prev_W1, Y1, prev_Y1, W2, prev_W2, Y2, prev_Y2) < tol:
            print("Convergence reached.")
            break
    # 计算字典矩阵
    D1 = train_data @ W1
    D2 = train_data @ W2
    # #保存收敛误差，以表格形式
    # convergence_error = np.array([W1_norms_e, W2_norms_e, Y1_norms_e, Y2_norms_e])
    # np.savetxt(f'convergence_error/d{i:02d}.txt', convergence_error)
    # #绘制收敛误差折线图,W1在左边，W2在右边
    # fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    # axs[0, 0].plot(W1_norms_e)
    # axs[0, 0].set_xlabel('Iteration')
    # axs[0, 0].set_ylabel('Norm Difference')
    # axs[0, 0].set_title('W1')
    # axs[0, 1].plot(W2_norms_e)
    # axs[0, 1].set_xlabel('Iteration')
    # axs[0, 1].set_ylabel('Norm Difference')
    # axs[0, 1].set_title('W2')
    # axs[1, 0].plot(Y1_norms_e)
    # axs[1, 0].set_xlabel('Iteration')
    # axs[1, 0].set_ylabel('Norm Difference')
    # axs[1, 0].set_title('Y1')
    # axs[1, 1].plot(Y2_norms_e)
    # axs[1, 1].set_xlabel('Iteration')
    # axs[1, 1].set_ylabel('Norm Difference')
    # axs[1, 1].set_title('Y2')
    # fig.tight_layout()
    # #plt.show()
    # # 保存图
    # plt.savefig(f'plts/convergence_error/d{i:02d}.png')

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

    #统计FDR,将每个故障的FAR记录下来,只计算统计量中的后800个变量
    T2_FDR=np.mean(T2_statistics[160:960] >= T2_limit)
    SPE_FDR=np.mean(SPE_statistics[160:960] >= SPE_limit)
    print(f"T2 FDR for d{i:02d}: {T2_FDR}")
    print(f"SPE FDR for d{i:02d}: {SPE_FDR}")
    #统计FAR
    T2_FAR = np.mean(T2_statistics[0:160] >= T2_limit)
    SPE_FAR = np.mean(SPE_statistics[0:160] >= SPE_limit)
    print(f"T2 FAR for d{i:02d}: {T2_FAR}")
    print(f"SPE FAR for d{i:02d}: {SPE_FAR}")

    # 将结果添加到列表中
    results.append({
        'file': f'd{i:02d}',
        'T2_FDR': T2_FDR,
        'SPE_FDR': SPE_FDR,
        'T2_FAR': T2_FAR,
        'SPE_FAR': SPE_FAR
    })


    # T2_FAR = np.mean(T2_statistics > T2_limit) # 统计FAR
    # SPE_FAR = np.mean(SPE_statistics > SPE_limit)

    # # 绘制 T² 统计量和控制限的折线图
    # plt.plot(T2_statistics, label='T2')
    # plt.axhline(y=T2_limit, color='r', linestyle='--', label='Control Limit')
    # plt.xlabel('Sample Index')
    # plt.ylabel('T2 Value')
    # plt.legend()
    # plt.title(f'T² Statistics for {test_file}')
    # #plt.show()
    # # 保存图，如果路不存在就新建路径
    # if not os.path.exists('TE_data/T2_statistics'):
    #     os.makedirs('TE_data/T2_statistics')
    # plt.savefig(f'TE_data/T2_statistics/d{i:02d}.png')
    #
    # # 绘制 SPE 统计量和控制限的折线图
    # plt.plot(SPE_statistics, label='SPE')
    # plt.axhline(y=SPE_limit, color='r', linestyle='--', label='Control Limit')
    # plt.xlabel('Sample Index')
    # plt.ylabel('SPE Value')
    # plt.legend() # 显示图例
    # plt.title(f'SPE Statistics for {test_file}')
    # #plt.show()
    # # 保存图，如果路不存在就新建路径
    # if not os.path.exists('TE_data/SPE_statistics'):
    #     os.makedirs('TE_data/SPE_statistics')
    # plt.savefig(f'TE_data/SPE_statistics/{test_file}.png')

    #绘制带有控制限的T2统计量和SPE统计量折线图，T2统计量在左边，SPE统计量在右边
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(T2_statistics, label='T2')
    axs[0].axhline(y=T2_limit, color='r', linestyle='--', label='Control Limit')
    axs[0].set_xlabel('Sample Index')
    axs[0].set_ylabel('T2 Value')
    axs[0].legend()
    axs[1].plot(SPE_statistics, label='SPE')
    axs[1].axhline(y=SPE_limit, color='r', linestyle='--', label='Control Limit')
    axs[1].set_xlabel('Sample Index')
    axs[1].set_ylabel('SPE Value')
    axs[1].legend()
    plt.suptitle(f'Statistics for {test_file}')
    fig.tight_layout()

    plt.savefig(f'plts/statistics/d{i:02d}.png')

    # #绘制字典热图,将两个字典绘制在同一幅图内，再将图保存，
    # plt.subplot(1, 2, 1)
    # plt.imshow(D1, cmap='hot')
    # plt.colorbar()
    # plt.subplot(1, 2, 2)
    # plt.imshow(D2, cmap='hot')
    # plt.colorbar()
    # plt.title(f'Dictionary for {train_file}')
    #
    # plt.savefig(f'plts/dictionary/d{i:02d}.png')
# 将所有结果写入CSV文件

import csv
with open('all_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['file', 'T2_FDR', 'SPE_FDR', 'T2_FAR', 'SPE_FAR']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for result in results:
        writer.writerow(result)
