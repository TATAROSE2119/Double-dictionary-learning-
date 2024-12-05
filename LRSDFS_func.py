import numpy as np

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

