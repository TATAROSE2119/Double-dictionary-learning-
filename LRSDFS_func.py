import numpy as np
from scipy.stats import gaussian_kde
# 定义逐个元素更新 W1 的函数
def update_W1(X, W1, W2, Y1, Y2, M, N, a):
    # 获取 W1 的形状
    rows, cols = W1.shape

    # 创建一个新的 W1 用于保存更新后的值
    W1_new = np.zeros_like(W1)

    # 分子部分 (X.T @ X @ Y1.T)
    numerator = X.T @ X @ Y1.T

    # 分母部分逐元素计算
    for i in range(rows):
        for j in range(cols):
            # 分母部分每个元素计算
            term1 = X.T @ X @ Y1 @ Y1.T  # 第一项
            term2 = X.T @ X @ W2 @ Y2 @ Y1.T  # 第二项
            term3 = a * X.T @ M @ N.T  # 第三项

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

    # 分母部分逐元素计算
    for i in range(rows):
        for j in range(cols):
            # 分母部分每个元素计算
            term1 = X.T @ X @ W1 @ Y1 @ Y2.T  # 第一项
            term2 = X.T @ X @ W2 @ Y2 @ Y2.T  # 第二项
            term3 = b * X.T @ np.outer(i_d, i_c.T) @ Y2.T  # 第三项

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

    # 分母部分逐元素计算
    for i in range(rows):
        for j in range(cols):
            # 分母部分每个元素计算
            term1 = W1.T @ X.T @ X @ W1 @ Y1  # 第一项
            term2 = W1.T @ X.T @ X @ W2 @ Y2  # 第二项
            term3 = c * U1 @ Y1  # 第三项
            term4 = 2 * e * Y1 @ Y1.T @ Y1  # 第四项

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

    # 分母部分逐元素计算
    for i in range(rows):
        for j in range(cols):
            # 分母部分每个元素计算
            term1 = W2.T @ X.T @ X @ W1 @ Y1  # 第一项
            term2 = W2.T @ X.T @ X @ W2 @ Y2  # 第二项
            term3 = b * W2.T @ X.T @ np.outer(i_d, i_c)  # 第三项
            term4 = d * U2 @ Y2  # 第四项
            term5 = 2 * f * Y2 @ Y2.T @ Y2  # 第五项

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

def calculate_feature_importance(Y1, Y2, p=0.5, q=0.5):
    """
    计算特征重要性
    :param Y1: 低秩编码矩阵 (k1 x n)
    :param Y2: 稀疏编码矩阵 (k2 x n)
    :param p: 权重参数，默认 0.5
    :param q: 权重参数，默认 0.5
    :return: 特征重要性列表
    """
    # 检查 p 和 q 是否满足条件
    assert p + q == 1, "p + q must be equal to 1"

    # 初始化特征重要性列表
    n_features = Y1.shape[1]  # 特征数目
    feature_importance = np.zeros(n_features)

    # 逐列计算特征重要性
    for i in range(n_features):
        # 计算 Y1 和 Y2 第 i 列的 L2 范数
        norm_Y1_i = np.linalg.norm(Y1[:, i])
        norm_Y2_i = np.linalg.norm(Y2[:, i])

        # 计算特征重要性
        feature_importance[i] = p * norm_Y1_i + q * norm_Y2_i

    return feature_importance

def select_top_features(train_data, eval_scores, l):
    """
    根据特征重要性得分选择前 l 个特征，生成新的数据矩阵 Xnew
    :param train_data: 原始训练数据矩阵 (480, 52)
    :param eval_scores: 特征重要性得分 (1, 52)
    :param l: 要选择的特征数量
    :return: 新的数据矩阵 Xnew
    """
    # 对特征重要性得分进行降序排序，获取排序后的索引
    sorted_indices = np.argsort(eval_scores)[::-1]  # 从大到小排序

    # 选择前 l 个特征的索引
    top_features_indices = sorted_indices[:l]

    # 根据选出的索引提取对应的列
    Xnew = train_data[:, top_features_indices]

    return Xnew, top_features_indices
def calculate_statistics(train_data,new_data, W1, W2, Y1, Y2, Sigma_Y1, Sigma_Y2):
    """
    计算新样本的 T² 和 SPE 统计量
    :param new_data: 新样本矩阵 (960, 52)
    :param W1: 低秩字典矩阵
    :param W2: 稀疏字典矩阵
    :param Y1: 训练阶段的低秩编码矩阵
    :param Y2: 训练阶段的稀疏编码矩阵
    :param Sigma_Y1: 低秩编码矩阵的协方差矩阵
    :param Sigma_Y2: 稀疏编码矩阵的协方差矩阵
    :return: T² 和 SPE 统计量列表
    """
    T2_statistics = []
    SPE_statistics = []

    # 协方差矩阵的逆
    Sigma_Y1_inv = np.linalg.inv(Sigma_Y1)
    Sigma_Y2_inv = np.linalg.inv(Sigma_Y2)

    for x_new in new_data:
        # Step 1: 标准化新样本（根据训练数据的均值和标准差）
        x_new_standardized = (x_new - train_data.mean(axis=0)) / train_data.std(axis=0)

        # Step 2: 计算编码矩阵 Y1 和 Y2
        y1_new = np.linalg.pinv(W1) @ x_new_standardized
        y2_new = np.linalg.pinv(W2) @ x_new_standardized

        # Step 3: 计算 T² 统计量
        T2 = (y1_new.T @ Sigma_Y1_inv @ y1_new) + (y2_new.T @ Sigma_Y2_inv @ y2_new)
        T2_statistics.append(T2)

        # Step 4: 计算 SPE 统计量
        reconstruction = W1 @ y1_new + W2 @ y2_new
        SPE = np.linalg.norm(x_new_standardized - reconstruction) ** 2
        SPE_statistics.append(SPE)

    return T2_statistics, SPE_statistics


def calculate_control_limit(statistics, percentile=99):
    """
    使用核密度估计 (KDE) 计算给定统计量的控制限。

    :param statistics: array-like, 样本的统计量数据
    :param percentile: float, 所需的分位数 (0-100)，默认为99
    :return: 控制限值
    """
    # 确保统计量为 NumPy 数组
    statistics = np.array(statistics)

    # 使用核密度估计拟合数据分布
    kde = gaussian_kde(statistics)

    # 生成统计量的范围，用于积分
    statistic_range = np.linspace(statistics.min(), statistics.max(), 1000)

    # 累积分布函数 (CDF)
    cdf = np.cumsum(kde(statistic_range))
    cdf /= cdf[-1]  # 归一化到 [0, 1]

    # 找到接近指定百分位数的统计值
    control_limit = statistic_range[np.searchsorted(cdf, percentile / 100.0)]

    return control_limit