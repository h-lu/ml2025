import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler

def generate_blob_data(n_samples=300, centers=4, cluster_std=0.8, random_state=42):
    """
    生成Blob形状的聚类数据
    
    参数:
    - n_samples: 样本总数
    - centers: 簇中心的数量
    - cluster_std: 每个簇的标准差(控制簇的分散程度)
    - random_state: 随机种子，保证结果可复现
    
    返回:
    - X: 特征矩阵(缩放后)
    - y: 真实簇标签
    """
    X, y = make_blobs(
        n_samples=n_samples, 
        centers=centers, 
        cluster_std=cluster_std, 
        random_state=random_state
    )
    
    # 特征缩放
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def generate_moon_data(n_samples=300, noise=0.05, random_state=42):
    """
    生成Moon形状的聚类数据
    
    参数:
    - n_samples: 样本总数
    - noise: 噪声程度
    - random_state: 随机种子，保证结果可复现
    
    返回:
    - X: 特征矩阵(缩放后)
    - y: 真实簇标签
    """
    X, y = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state
    )
    
    # 特征缩放
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def generate_varied_blobs(n_samples=500, random_state=42):
    """
    生成密度不同的Blob数据，用于展示DBSCAN的优势
    
    参数:
    - n_samples: 样本总数
    - random_state: 随机种子
    
    返回:
    - X: 特征矩阵(缩放后)
    - y: 真实簇标签
    """
    rng = np.random.RandomState(random_state)
    
    # 生成第一个紧密的簇
    X1, y1 = make_blobs(
        n_samples=int(0.3 * n_samples),
        centers=[[0, 0]],
        cluster_std=0.3,
        random_state=random_state
    )
    y1 = np.zeros(X1.shape[0], dtype=int)
    
    # 生成第二个稀疏的簇
    X2, y2 = make_blobs(
        n_samples=int(0.4 * n_samples),
        centers=[[3, 3]],
        cluster_std=0.8,
        random_state=random_state
    )
    y2 = np.ones(X2.shape[0], dtype=int)
    
    # 生成第三个不规则形状的簇
    X3 = rng.normal(size=(int(0.3 * n_samples), 2)) * 0.5
    X3[:, 0] += -2
    X3[:, 1] += -2
    y3 = np.full(X3.shape[0], 2, dtype=int)
    
    # 合并数据
    X = np.vstack([X1, X2, X3])
    y = np.hstack([y1, y2, y3])
    
    # 特征缩放
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y 