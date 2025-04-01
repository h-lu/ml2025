import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles

def generate_blob_data(n_samples=300, n_centers=3, random_state=42):
    """
    生成用于聚类的球状数据
    
    参数:
        n_samples: 样本数量
        n_centers: 簇的数量
        random_state: 随机种子
        
    返回:
        X: 生成的特征数据 shape=(n_samples, 2)
        y: 真实的簇标签 shape=(n_samples,)
    """
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=n_centers,
        cluster_std=0.8,
        random_state=random_state
    )
    return X, y

def generate_moons_data(n_samples=300, noise=0.1, random_state=42):
    """
    生成月牙形状的数据
    
    参数:
        n_samples: 样本数量
        noise: 噪声水平
        random_state: 随机种子
        
    返回:
        X: 生成的特征数据 shape=(n_samples, 2)
        y: 真实的簇标签 shape=(n_samples,)
    """
    X, y = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state
    )
    return X, y

def generate_circles_data(n_samples=300, noise=0.05, factor=0.5, random_state=42):
    """
    生成环形数据
    
    参数:
        n_samples: 样本数量
        noise: 噪声水平
        factor: 内圆与外圆的比例
        random_state: 随机种子
        
    返回:
        X: 生成的特征数据 shape=(n_samples, 2)
        y: 真实的簇标签 shape=(n_samples,)
    """
    X, y = make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        random_state=random_state
    )
    return X, y

def generate_custom_data():
    """
    生成用于手动 K-means 互动的小数据集
    
    返回:
        X: 生成的特征数据
    """
    return np.array([
        [1, 1],  # A
        [1, 2],  # B
        [2, 1],  # C
        [5, 4],  # D
        [5, 5],  # E
        [6, 5]   # F
    ])

def generate_anisotropic_data(n_samples=300, n_centers=3, random_state=42):
    """
    生成不同大小和密度的球状数据
    
    参数:
        n_samples: 样本数量
        n_centers: 簇的数量
        random_state: 随机种子
        
    返回:
        X: 生成的特征数据 shape=(n_samples, 2)
        y: 真实的簇标签 shape=(n_samples,)
    """
    np.random.seed(random_state)
    C1 = np.random.randn(n_samples//3, 2) * 0.5 + np.array([2, 2])
    C2 = np.random.randn(n_samples//3, 2) * 1.5 + np.array([-3, -3])
    C3 = np.random.randn(n_samples//3, 2) * 1.0 + np.array([5, -2])
    
    X = np.vstack([C1, C2, C3])
    y = np.hstack([
        np.zeros(n_samples//3),
        np.ones(n_samples//3),
        np.ones(n_samples//3) * 2
    ])
    
    # 打乱数据
    idx = np.random.permutation(len(X))
    return X[idx], y[idx] 