import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from utils.helpers import generate_ecommerce_data

def load_dataset(dataset_name, test_size=0.2, random_state=42):
    """
    加载和划分数据集
    
    参数:
    dataset_name: 数据集名称，可选值：'iris', 'wine', 'breast_cancer', 'ecommerce'
    test_size: 测试集大小比例
    random_state: 随机种子
    
    返回:
    X_train, X_test, y_train, y_test, feature_names, target_names
    """
    if dataset_name == 'iris':
        data = load_iris()
        X, y = data.data, data.target
        feature_names = data.feature_names
        target_names = data.target_names
        
    elif dataset_name == 'wine':
        data = load_wine()
        X, y = data.data, data.target
        feature_names = data.feature_names
        target_names = data.target_names
        
    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer()
        X, y = data.data, data.target
        feature_names = data.feature_names
        target_names = data.target_names
        
    elif dataset_name == 'ecommerce':
        df = generate_ecommerce_data(n_samples=1000, random_state=random_state)
        X = df.drop(['user_id', 'high_value_user'], axis=1)
        y = df['high_value_user']
        feature_names = X.columns.tolist()
        target_names = ['低价值用户', '高价值用户']
        
    else:
        raise ValueError(f"未知数据集: {dataset_name}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, feature_names, target_names

def load_from_csv(file_path, target_column, test_size=0.2, random_state=42):
    """
    从CSV文件加载数据集
    
    参数:
    file_path: CSV文件路径
    target_column: 目标变量列名
    test_size: 测试集大小比例
    random_state: 随机种子
    
    返回:
    X_train, X_test, y_train, y_test, feature_names
    """
    df = pd.read_csv(file_path)
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    feature_names = X.columns.tolist()
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, feature_names 