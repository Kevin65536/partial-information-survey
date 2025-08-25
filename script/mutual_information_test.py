import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
from itertools import product
import pandas as pd

def discrete_entropy(x):
    """计算离散变量的熵"""
    _, counts = np.unique(x, return_counts=True)
    probs = counts / len(x)
    return -np.sum(probs * np.log2(probs + 1e-12))

def discrete_mutual_info(x, y):
    """计算离散变量的互信息"""
    # 转换为字符串以确保正确处理
    x_str = [str(val) for val in x]
    y_str = [str(val) for val in y]
    return mutual_info_score(x_str, y_str)

def conditional_mutual_info(x, y, z):
    """计算条件互信息 I(X;Y|Z)"""
    # 对于每个Z的取值，计算条件互信息
    z_values = np.unique(z)
    cmi = 0
    n_total = len(z)
    
    for z_val in z_values:
        mask = z == z_val
        if np.sum(mask) > 0:
            x_given_z = x[mask]
            y_given_z = y[mask]
            p_z = np.sum(mask) / n_total
            if len(x_given_z) > 1:  # 确保有足够的样本
                mi_given_z = discrete_mutual_info(x_given_z, y_given_z)
                cmi += p_z * mi_given_z
    
    return cmi

def multivariate_mutual_info(x1, x2, y):
    """计算多变量互信息 I(X1,X2;Y)"""
    # 将X1和X2组合成联合变量
    x_joint = np.array([f"{v1},{v2}" for v1, v2 in zip(x1, x2)])
    y_str = [str(val) for val in y]
    return mutual_info_score(x_joint, y_str)

def triple_mutual_info(x1, x2, y):
    """计算三变量互信息 I(X1;X2;Y)"""
    # 使用公式: I(X1;X2;Y) = I(X1;Y) + I(X2;Y) + I(X1;X2) - I(X1,X2;Y)
    i_x1_x2 = discrete_mutual_info(x1, x2)
    i_x1_y = discrete_mutual_info(x1, y)
    i_x2_y = discrete_mutual_info(x2, y)
    i_x1x2_y = multivariate_mutual_info(x1, x2, y)
    
    result = i_x1_y + i_x2_y + i_x1_x2 - i_x1x2_y
    
    print(f"  三变量互信息计算详情:")
    print(f"  I(X1;Y) = {i_x1_y:.6f}")
    print(f"  I(X2;Y) = {i_x2_y:.6f}")
    print(f"  I(X1;X2) = {i_x1_x2:.6f}")
    print(f"  I(X1,X2;Y) = {i_x1x2_y:.6f}")
    print(f"  I(X1;X2;Y) = {i_x1_y:.6f} + {i_x2_y:.6f} + {i_x1_x2:.6f} - {i_x1x2_y:.6f} = {result:.6f}")
    
    return result

def conditional_entropy(y, x):
    """计算条件熵 H(Y|X)"""
    h_y = discrete_entropy(y)
    mi_xy = discrete_mutual_info(x, y)
    return h_y - mi_xy

# 生成完全独立的X1, X2和Y = X1 XOR X2
np.random.seed(42)  # 设置随机种子以获得可重复的结果
n_samples = 100000

# 生成完全独立的X1和X2（均匀分布的二元变量）
X1 = np.random.randint(0, 2, size=n_samples)
X2 = np.random.randint(0, 2, size=n_samples)

# Y = X1 XOR X2
Y = X1 ^ X2

print("数据生成完成:")
print(f"样本数量: {n_samples}")
print(f"X1的分布: {np.bincount(X1) / n_samples}")
print(f"X2的分布: {np.bincount(X2) / n_samples}")
print(f"Y的分布: {np.bincount(Y) / n_samples}")
print()

# 验证X1和X2的独立性
contingency_table = pd.crosstab(X1, X2, normalize=True)
print("X1和X2的联合分布（验证独立性）:")
print(contingency_table)
print()

# 计算各种互信息
print("互信息计算结果:")
print("=" * 50)

# 1. I(X1; X2) - 应该约等于0（因为X1和X2完全独立）
i_x1_x2 = discrete_mutual_info(X1, X2)
print(f"I(X1; X2) = {i_x1_x2:.6f}")

# 2. I(X1; Y) - X1和Y之间的互信息
i_x1_y = discrete_mutual_info(X1, Y)
print(f"I(X1; Y) = {i_x1_y:.6f}")

# 3. I(X2; Y) - X2和Y之间的互信息
i_x2_y = discrete_mutual_info(X2, Y)
print(f"I(X2; Y) = {i_x2_y:.6f}")

# 4. I(X1,X2; Y) - 联合信息X1,X2与Y之间的互信息
i_x1x2_y = multivariate_mutual_info(X1, X2, Y)
print(f"I(X1,X2; Y) = {i_x1x2_y:.6f}")

# 5. I(X1; X2; Y) - 三变量互信息
i_x1_x2_y_triple = triple_mutual_info(X1, X2, Y)
print(f"I(X1; X2; Y) = {i_x1_x2_y_triple:.6f}")

# 6. I(X1; X2|Y) - 条件互信息X1和X2在给定Y条件下的互信息
i_x1_x2_given_y = conditional_mutual_info(X1, X2, Y)
print(f"I(X1; X2|Y) = {i_x1_x2_given_y:.6f}")

# 条件熵验证
h_y_given_x1 = conditional_entropy(Y, X1)
h_y_given_x2 = conditional_entropy(Y, X2)

print()
print("条件熵验证:")
print(f"H(Y|X1) = {h_y_given_x1:.6f} ≈ 1 (已知X1后Y仍有最大不确定性)")
print(f"H(Y|X2) = {h_y_given_x2:.6f} ≈ 1 (已知X2后Y仍有最大不确定性)")

# 计算H(X1),H(X2),H(Y),H(X1,X2),H(X1,Y),H(X2,Y),H(X1,X2,Y)

h_x1 = discrete_entropy(X1)
h_x2 = discrete_entropy(X2)
h_y = discrete_entropy(Y)
h_x1x2 = discrete_entropy(np.array([f"{v1},{v2}" for v1, v2 in zip(X1, X2)]))
h_x1y = discrete_entropy(np.array([f"{v1},{v2}" for v1, v2 in zip(X1, Y)]))
h_x2y = discrete_entropy(np.array([f"{v1},{v2}" for v1, v2 in zip(X2, Y)]))
h_x1x2y = discrete_entropy(np.array([f"{v1},{v2},{v3}" for v1, v2, v3 in zip(X1, X2, Y)]))

print(f"H(X1,X2,Y) = {h_x1x2y:.6f}")
print(f"H(X1,Y) = {h_x1y:.6f}")
print(f"H(X2,Y) = {h_x2y:.6f}")
print(f"H(X1,X2) = {h_x1x2:.6f}")

print(f"H(X1,X2,Y) - H(X1,Y) - H(X2,Y) - H(X1,X2) + H(X1) + H(X2) + H(Y) = {h_x1x2y - h_x1y - h_x2y - h_x1x2 + h_x1 + h_x2 + h_y:.6f}")