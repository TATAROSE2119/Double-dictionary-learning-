import numpy as np
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import Lasso
# 定义逐个元素更新 W1 的函数
def update_W1(X, W1, W2, Y1, Y2, M, N, a):
    # 获取 W1 的形状
    rows, cols = W1.shape

    # 创建一个新的 W1 用于保存更新后的值
    W1_new = np.zeros_like(W1)

    # 分子部分 (X.T @ X @ Y1.T)
    numerator = X.T @ X @ Y1.T
    # 分母部分每个元素计算
    term1 = X.T @ X @W1 @ Y1 @ Y1.T  # 第一项
    term2 = X.T @ X @ W2 @ Y2 @ Y1.T  # 第二项
    term3 = a * X.T @ M @ N.T  # 第三项

    # 分母部分逐元素计算
    for i in range(rows):
        for j in range(cols):

            denominator = term1[i, j] + term2[i, j] + term3[i, j]

            # 更新 W1 的元素
            if denominator > 1e-8:  # 防止除以零
                W1_new[i, j] = numerator[i, j] / denominator
            else:
                W1_new[i, j] = 0.0

    return W1_new

# 定义逐个元素更新 W2 的函数
def update_W2(X, W1, W2, Y1, Y2, i_d, i_c, b):
    # 获取 W2 的形状
    rows, cols = W2.shape

    # 创建一个新的 W2 用于保存更新后的值
    W2_new = np.zeros_like(W2)

    # 分子部分 (X.T @ X @ Y2.T)
    numerator = X.T @ X @ Y2.T
    # 分母部分每个元素计算
    term1 = X.T @ X @ W1 @ Y1 @ Y2.T  # 第一项
    term2 = X.T @ X @ W2 @ Y2 @ Y2.T  # 第二项
    term3 = b * X.T @ np.outer(i_d, i_c.T) @ Y2.T  # 第三项

    # 分母部分逐元素计算
    for i in range(rows):
        for j in range(cols):

            denominator = term1[i, j] + term2[i, j] + term3[i, j]

            # 更新 W2 的元素
            if denominator > 1e-8:  # 防止除以零
                W2_new[i, j] = numerator[i, j] / denominator
            else:
                W2_new[i, j] = 0.0

    return W2_new

# 定义逐个元素更新 Y1 的函数
def update_Y1(X, W1, W2, Y1, Y2, U1, c, e):
    # 获取 Y1 的形状
    rows, cols = Y1.shape

    # 创建一个新的 Y1 用于保存更新后的值
    Y1_new = np.zeros_like(Y1)

    # 分子部分: W1^T @ X^T @ X + 2eY1
    numerator = W1.T @ X.T @ X + 2 * e * Y1
    # 分母部分每个元素计算
    term1 = W1.T @ X.T @ X @ W1 @ Y1  # 第一项
    term2 = W1.T @ X.T @ X @ W2 @ Y2  # 第二项
    term3 = c * U1 @ Y1  # 第三项
    term4 = 2 * e * Y1 @ Y1.T @ Y1  # 第四项

    # 分母部分逐元素计算
    for i in range(rows):
        for j in range(cols):

            denominator = term1[i, j] + term2[i, j] + term3[i, j] + term4[i, j]

            # 更新 Y1 的元素
            if denominator > 1e-8:  # 防止除以零
                Y1_new[i, j] = numerator[i, j] / denominator
            else:
                Y1_new[i, j] = 0.0

    return Y1_new

# 定义逐个元素更新 Y2 的函数
def update_Y2(X, W1, W2, Y1, Y2, U2, i_d, i_c, b, d, f):
    # 获取 Y2 的形状
    rows, cols = Y2.shape

    # 创建一个新的 Y2 用于保存更新后的值
    Y2_new = np.zeros_like(Y2)

    # 分子部分: W2^T @ X^T @ X + 2fY2
    numerator = W2.T @ X.T @ X + 2 * f * Y2
    # 分母部分每个元素计算
    term1 = W2.T @ X.T @ X @ W1 @ Y1  # 第一项
    term2 = W2.T @ X.T @ X @ W2 @ Y2  # 第二项
    term3 = b * W2.T @ X.T @ np.outer(i_d, i_c)  # 第三项
    term4 = d * U2 @ Y2  # 第四项
    term5 = 2 * f * Y2 @ Y2.T @ Y2  # 第五项

    # 分母部分逐元素计算
    for i in range(rows):
        for j in range(cols):

            denominator = term1[i, j] + term2[i, j] + term3[i, j] + term4[i, j] + term5[i, j]

            # 更新 Y2 的元素
            if denominator > 1e-8:  # 防止除以零
                Y2_new[i, j] = numerator[i, j] / denominator
            else:
                Y2_new[i, j] = 0.0

    return Y2_new

# 定义更新 U1 和 U2 的函数
def update_U(Y1, Y2):
    # 初始化 U1 和 U2
    U1 = np.zeros((Y1.shape[0], Y1.shape[0]))
    U2 = np.zeros((Y2.shape[0], Y2.shape[0]))

    # 更新 U1 的对角元素
    for i in range(Y1.shape[0]):
        norm_Y1_i = np.linalg.norm(Y1[i, :])  # 计算第 i 行的 L2 范数
        if norm_Y1_i > 1e-8:  # 避免除以 0
            U1[i, i] = 1 / (2 * norm_Y1_i)
        else:
            U1[i, i] = 0.0

    # 更新 U2 的对角元素
    for i in range(Y2.shape[0]):
        norm_Y2_i = np.linalg.norm(Y2[i, :])  # 计算第 i 行的 L2 范数
        if norm_Y2_i > 1e-8:  # 避免除以 0
            U2[i, i] = 1 / (2 * norm_Y2_i)
        else:
            U2[i, i] = 0.0

    return U1, U2

#计算统计量
def calculate_statistics(X_new, D1, D2, a=0.5, b=0.5):
    """
    根据第二种方法逐个样本计算 T² 和 SPE 统计量
    :param X_new: 新数据矩阵 (n_features, n_samples)
    :param D1: 低秩字典矩阵
    :param D2: 稀疏字典矩阵
    :param a: T² 统计量中低秩部分的权重
    :param b: T² 统计量中稀疏部分的权重
    :return: T² 和 SPE 统计量列表
    """
    n_samples = X_new.shape[1]  # 样本数
    T2_statistics = []
    SPE_statistics = []

    for i in range(n_samples):
        # 获取单个样本 x
        x = X_new[:, i].reshape(-1, 1)  # 转为列向量

        # 计算编码矩阵的估计值
        Y1_hat = np.linalg.pinv(D1.T @ D1) @ D1.T @ x
        Y2_hat = np.linalg.pinv(D2.T @ D2) @ D2.T @ x

        # 重构样本
        X_new_hat = D1 @ Y1_hat + D2 @ Y2_hat

        # 计算 T² 统计量
        T2 = a * Y1_hat.T  @  Y1_hat + b * Y2_hat.T @ Y2_hat
        T2_statistics.append(T2.item())  # 转为标量

        # 计算 SPE 统计量
        SPE = (x - X_new_hat).T @ (x - X_new_hat)
        SPE_statistics.append(SPE.item())

    return np.array(T2_statistics), np.array(SPE_statistics)

# 计算控制限
def calculate_control_limit_kde(statistics, bandwidth=0.2, percentile=99):
    """
    使用 KDE 方法计算统计量的控制限
    :param statistics: 输入的统计量数组 (一维数组)
    :param bandwidth: KDE 的带宽参数
    :param percentile: 控制限的百分位 (默认为 99)
    :return: 控制限值
    """
    # 将统计量转为二维列向量
    statistics = np.array(statistics).reshape(-1, 1)

    # 使用 KDE 拟合数据
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(statistics)

    # 生成估计分布的概率密度值
    density = np.exp(kde.score_samples(statistics))

    # 计算指定百分位数的控制限
    control_limit = np.percentile(statistics, percentile)
    return control_limit