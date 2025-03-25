import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def generate_synthetic_data(n_samples=1000, noise_level=1.0, random_state=42):
    """
    生成合成的非线性数据用于XGBoost演示
    
    参数:
    n_samples: int, 样本数量
    noise_level: float, 噪声水平
    random_state: int, 随机种子
    
    返回:
    X: 特征矩阵
    y: 目标向量
    """
    np.random.seed(random_state)
    X = np.random.rand(n_samples, 5)
    # 创建非线性关系
    y = 5 + 3*X[:, 0]**2 + 2*X[:, 1] + np.sin(3*X[:, 2]) + np.exp(X[:, 3]) - 2*X[:, 4] + np.random.normal(0, noise_level, n_samples)
    
    return X, y

def generate_complex_data(n_samples=1000, random_state=42):
    """
    生成更复杂的合成数据，包含交互特征和非线性关系
    
    参数:
    n_samples: int, 样本数量
    random_state: int, 随机种子
    
    返回:
    X: 特征矩阵
    y: 目标向量
    """
    np.random.seed(random_state)
    X = np.random.rand(n_samples, 6)
    # 创建复杂的非线性关系
    y = (
        2 + 
        3 * X[:, 0]**2 + 
        2 * np.sin(2 * np.pi * X[:, 1]) + 
        4 * (X[:, 2] - 0.5)**2 + 
        X[:, 3] * X[:, 4] +  # 交互特征
        1.5 * np.exp(X[:, 5]) + 
        np.random.normal(0, 0.5, n_samples)  # 噪声
    )
    
    return X, y

def split_and_preprocess_data(X, y, test_size=0.2, random_state=42, scale=True):
    """
    分割数据并进行预处理
    
    参数:
    X: 特征矩阵
    y: 目标向量
    test_size: float, 测试集比例
    random_state: int, 随机种子
    scale: bool, 是否进行标准化
    
    返回:
    X_train_processed: 处理后的训练特征
    X_test_processed: 处理后的测试特征
    y_train: 训练目标
    y_test: 测试目标
    """
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    if scale:
        # 标准化数据
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train)
        X_test_processed = scaler.transform(X_test)
    else:
        X_train_processed = X_train
        X_test_processed = X_test
        
    return X_train_processed, X_test_processed, y_train, y_test

def load_housing_data():
    """
    加载房价数据集（如果可用）
    
    返回:
    df: 数据DataFrame
    X: 特征矩阵
    y: 目标向量
    feature_names: 特征名称列表
    """
    try:
        # 尝试加载sklearn的California房价数据集
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing()
        
        X = housing.data
        y = housing.target
        feature_names = housing.feature_names
        
        # 创建DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['PRICE'] = y
        
        return df, X, y, feature_names
    except:
        print("无法加载California房价数据集。使用合成数据替代。")
        # 生成合成数据
        X, y = generate_synthetic_data(1000)
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        df = pd.DataFrame(X, columns=feature_names)
        df['price'] = y
        
        return df, X, y, feature_names

def create_preprocessing_pipeline(df, target_column):
    """
    创建特征预处理管道
    
    参数:
    df: 数据DataFrame
    target_column: 目标列名称
    
    返回:
    preprocessor: 预处理管道
    """
    # 分离特征和目标
    X = df.drop(target_column, axis=1)
    
    # 识别数值和类别特征
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # 创建预处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features if len(categorical_features) > 0 else [])
        ],
        remainder='passthrough'
    )
    
    return preprocessor 