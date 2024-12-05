import numpy as np
import LRSDFS_func
# Load train data
train_data = np.loadtxt('TE_data/train_data/d01.dat')
# Load test data
test_data = np.loadtxt('TE_data/test_data/d01_te.dat')

#进行数据的标准化
train_data = (train_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0) # axis=0表示按列进行操作
test_data = (test_data - np.mean(test_data, axis=0)) / np.std(test_data, axis=0) # axis=0表示按列进行操作
#转置
#train_data = train_data.T
#test_data = test_data.T
#print the shape of train_data
print(train_data.shape)
#训练数据的行数和列数

# 初始化字典矩阵 W1 和 W2（假设字典的原子数目为 k1 和 k2）
k1 = 52  # 低秩字典的原子数量
k2 = 52  # 稀疏字典的原子数量
W1 = np.random.rand(52, k1)  # 低秩字典矩阵 (52个特征, k1个字典原子)
W2 = np.random.rand(52, k2)  # 稀疏字典矩阵 (52个特征, k2个字典原子)

# 初始化编码矩阵 Y1 和 Y2（编码矩阵的维度与数据集 X 对应）
Y1 = np.random.rand(k1, 52)  # 低秩编码矩阵 (k1个字典原子, 52)
Y2 = np.random.rand(k2, 52)  # 稀疏编码矩阵 (k2个字典原子, 52)

# 初始化用于奇异值分解的矩阵 U1 和 U2
U1 = np.random.rand(k1, k1)  # 用于低秩字典学习的 SVD 矩阵
U2 = np.random.rand(k2, k2)  # 用于稀疏字典学习的 SVD 矩阵

# 向量 i_d 和 i_c
i_d = np.ones(480)
i_c = np.ones(52)

# 打印初始化结果的形状，确保一切正确
print(f"W1 shape: {W1.shape}")
print(f"W2 shape: {W2.shape}")
print(f"Y1 shape: {Y1.shape}")
print(f"Y2 shape: {Y2.shape}")
print(f"U1 shape: {U1.shape}")
print(f"U2 shape: {U2.shape}")
print(f"i_d shape: {i_d.shape}")
print(f"i_c shape: {i_c.shape}")

#对XW_1 SVD
XW_1 = train_data @ W1
M, S1, N = np.linalg.svd(XW_1, full_matrices=False)
N=N.T
#updata W1


# 更新 W1
W1_updated = LRSDFS_func.update_W1(X=train_data, W1=W1, W2=W2, Y1=Y1, Y2=Y2, M=M, N=N, a=0.5)

#updata W2


# 更新 W2
b = 0.5  # 假设的参数 b
W2_updated = LRSDFS_func.update_W2(X=train_data, W1=W1, W2=W2, Y1=Y1, Y2=Y2, i_d=i_d, i_c=i_c, b=b)
#updata Y1

# 更新 Y1
c = 0.5  # 假设的参数 c
e = 0.1  # 假设的参数 e
Y1_updated = LRSDFS_func.update_Y1(X=train_data, W1=W1, W2=W2, Y1=Y1, Y2=Y2, U1=U1, c=c, e=e)
#updata Y2

# 更新 Y2
b = 0.5  # 假设的参数 b
d = 0.1  # 假设的参数 d
f = 0.2  # 假设的参数 f
Y2_updated = LRSDFS_func.update_Y2(X=train_data, W1=W1, W2=W2, Y1=Y1, Y2=Y2, U2=U2, i_d=i_d, i_c=i_c, b=b, d=d, f=f)

#updata U1 U2

# 更新 U1 和 U2
U1_updated, U2_updated = LRSDFS_func.update_U(Y1=Y1_updated, Y2=Y2_updated)

