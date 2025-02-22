import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS
plt.rcParams['axes.unicode_minus'] = False

def plot_missing_values():
    """使用missingno库创建缺失值可视化"""
    # 创建包含缺失值的示例数据
    np.random.seed(42)
    n_samples = 100
    df = pd.DataFrame({
        '用户ID': range(n_samples),
        '年龄': np.random.randint(18, 70, n_samples),
        '收入': np.random.normal(10000, 2000, n_samples)
    })
    
    # 随机添加缺失值
    df.loc[np.random.choice(n_samples, 20), '年龄'] = np.nan
    df.loc[np.random.choice(n_samples, 30), '收入'] = np.nan
    
    # 创建缺失值矩阵图
    plt.figure(figsize=(10, 6))
    msno.matrix(df)
    plt.title('缺失值分布矩阵图')
    plt.tight_layout()
    plt.savefig('img/week2/missing_values_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建缺失值相关性热力图
    plt.figure(figsize=(8, 6))
    msno.heatmap(df)
    plt.title('缺失值相关性热力图')
    plt.tight_layout()
    plt.savefig('img/week2/missing_values_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_boxplot_outliers():
    """创建箱线图异常值检测示例"""
    # 生成包含异常值的数据
    np.random.seed(42)
    normal_data = np.random.normal(100, 15, 100)
    outliers = np.array([160, 170, 30, 20])
    data = np.concatenate([normal_data, outliers])
    
    # 创建箱线图
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data)
    plt.title('箱线图异常值检测')
    plt.ylabel('值')
    plt.savefig('img/week2/boxplot_outliers.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_scaling_comparison():
    """创建标准化和归一化效果对比图"""
    # 生成示例数据
    np.random.seed(42)
    data = np.random.normal(100, 20, 100)
    
    # 标准化和归一化
    scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()
    
    data_standardized = scaler.fit_transform(data.reshape(-1, 1))
    data_normalized = minmax_scaler.fit_transform(data.reshape(-1, 1))
    
    # 创建对比图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始数据分布
    sns.histplot(data=data, ax=ax1, bins=20)
    ax1.set_title('原始数据分布')
    
    # 标准化后的分布
    sns.histplot(data=data_standardized, ax=ax2, bins=20)
    ax2.set_title('标准化后的分布')
    
    # 归一化后的分布
    sns.histplot(data=data_normalized, ax=ax3, bins=20)
    ax3.set_title('归一化后的分布')
    
    plt.tight_layout()
    plt.savefig('img/week2/scaling_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_polynomial_features():
    """创建多项式特征转换效果图"""
    # 生成示例数据
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y = 0.5 * X**2 + X + 2 + np.random.normal(0, 0.2, (100, 1))
    
    # 多项式特征转换
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # 绘制原始数据和多项式特征
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.5)
    plt.title('原始数据')
    plt.xlabel('X')
    plt.ylabel('y')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_poly[:, 0], y, alpha=0.5, label='X')
    plt.scatter(X_poly[:, 1], y, alpha=0.5, label='X²')
    plt.title('多项式特征转换')
    plt.xlabel('特征值')
    plt.ylabel('y')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('img/week2/polynomial_features.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_log_transform():
    """创建对数变换效果对比图"""
    # 生成长尾分布数据
    np.random.seed(42)
    data = np.exp(np.random.normal(0, 1, 1000))
    
    # 对数变换
    data_log = np.log1p(data)
    
    # 创建对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 原始数据分布
    sns.histplot(data=data, ax=ax1, bins=50)
    ax1.set_title('原始数据分布（长尾分布）')
    
    # 对数变换后的分布
    sns.histplot(data=data_log, ax=ax2, bins=50)
    ax2.set_title('对数变换后的分布')
    
    plt.tight_layout()
    plt.savefig('img/week2/log_transform.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # 生成所有图
    plot_missing_values()
    plot_boxplot_outliers()
    plot_scaling_comparison()
    plot_polynomial_features()
    plot_log_transform() 