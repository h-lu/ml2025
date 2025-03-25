import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import xgboost as xgb

def train_linear_model(X_train, y_train):
    """
    训练线性回归模型
    
    参数:
    X_train: 训练特征
    y_train: 训练目标
    
    返回:
    model: 训练好的线性回归模型
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_polynomial_model(X_train, y_train, degree=2):
    """
    训练多项式回归模型
    
    参数:
    X_train: 训练特征
    y_train: 训练目标
    degree: 多项式阶数
    
    返回:
    model: 训练好的多项式回归模型
    """
    # 创建多项式特征
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_train)
    
    # 训练线性回归
    model = LinearRegression()
    model.fit(X_poly, y_train)
    
    # 返回管道
    return Pipeline([
        ('poly', poly),
        ('linear', model)
    ])

def train_xgboost_model(X_train, y_train, params=None):
    """
    训练XGBoost模型
    
    参数:
    X_train: 训练特征
    y_train: 训练目标
    params: 模型参数字典
    
    返回:
    model: 训练好的XGBoost模型
    """
    if params is None:
        # 默认参数
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': 42
        }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name="模型"):
    """
    评估模型性能
    
    参数:
    model: 训练好的模型
    X_test: 测试特征
    y_test: 测试目标
    model_name: 模型名称
    
    返回:
    metrics: 包含各种评估指标的字典
    """
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算各种指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 打印结果
    print(f"{model_name} 评估结果:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    # 返回指标字典
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'y_pred': y_pred
    }

def cross_validate_model(model, X, y, cv=5, scoring='neg_mean_squared_error'):
    """
    交叉验证模型性能
    
    参数:
    model: 模型
    X: 特征
    y: 目标
    cv: 交叉验证折数
    scoring: 评分指标
    
    返回:
    cv_scores: 交叉验证分数
    mean_score: 平均分数
    """
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    if 'neg_mean_squared_error' in scoring:
        # 转换为RMSE
        cv_scores = np.sqrt(-cv_scores)
        mean_score = np.mean(cv_scores)
        print(f"交叉验证RMSE: {cv_scores}")
        print(f"平均RMSE: {mean_score:.4f}")
    else:
        mean_score = np.mean(cv_scores)
        print(f"交叉验证分数: {cv_scores}")
        print(f"平均分数: {mean_score:.4f}")
    
    return cv_scores, mean_score

def compare_models(models, X_test, y_test, names=None):
    """
    比较多个模型的性能
    
    参数:
    models: 模型列表
    X_test: 测试特征
    y_test: 测试目标
    names: 模型名称列表
    
    返回:
    results: 包含各模型评估结果的字典
    """
    if names is None:
        names = [f"模型{i+1}" for i in range(len(models))]
    
    results = {}
    for model, name in zip(models, names):
        results[name] = evaluate_model(model, X_test, y_test, name)
    
    # 可视化比较
    plot_model_comparison(results, names)
    
    return results

def plot_model_comparison(results, names):
    """
    可视化模型性能比较
    
    参数:
    results: 包含各模型评估结果的字典
    names: 模型名称列表
    """
    # 提取RMSE和R²
    rmse_values = [results[name]['rmse'] for name in names]
    r2_values = [results[name]['r2'] for name in names]
    
    # 创建比较图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # RMSE比较
    axes[0].bar(names, rmse_values, color='skyblue')
    axes[0].set_title('RMSE比较 (越低越好)')
    axes[0].set_ylabel('RMSE')
    axes[0].tick_params(axis='x', rotation=45)
    
    # R²比较
    axes[1].bar(names, r2_values, color='lightgreen')
    axes[1].set_title('R²比较 (越高越好)')
    axes[1].set_ylabel('R²')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names=None):
    """
    可视化XGBoost模型的特征重要性
    
    参数:
    model: XGBoost模型
    feature_names: 特征名称列表
    
    返回:
    fig: 特征重要性图形
    """
    if not isinstance(model, xgb.XGBRegressor) and not isinstance(model, xgb.XGBClassifier):
        print("错误：此函数只适用于XGBoost模型")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制特征重要性
    if feature_names is not None:
        xgb.plot_importance(model, ax=ax, feature_names=feature_names)
    else:
        xgb.plot_importance(model, ax=ax)
    
    plt.title('特征重要性')
    plt.tight_layout()
    return fig

def plot_predictions(y_test, y_pred, title="预测值 vs 实际值"):
    """
    可视化预测结果
    
    参数:
    y_test: 实际值
    y_pred: 预测值
    title: 图表标题
    
    返回:
    fig: 预测对比图
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制散点图
    ax.scatter(y_test, y_pred, alpha=0.5)
    
    # 添加45°线
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    ax.set_xlabel('实际值')
    ax.set_ylabel('预测值')
    ax.set_title(title)
    
    # 添加性能指标
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    ax.text(0.05, 0.95, f'RMSE: {rmse:.4f}\nR²: {r2:.4f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_learning_curves(model, X_train, y_train, X_test, y_test, train_sizes=None):
    """
    绘制学习曲线
    
    参数:
    model: 模型
    X_train: 训练特征
    y_train: 训练目标
    X_test: 测试特征
    y_test: 测试目标
    train_sizes: 训练集大小比例
    
    返回:
    fig: 学习曲线图
    """
    from sklearn.model_selection import learning_curve
    
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    # 计算学习曲线
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=train_sizes,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    # 转换为RMSE
    train_rmse = np.sqrt(-np.mean(train_scores, axis=1))
    test_rmse = np.sqrt(-np.mean(test_scores, axis=1))
    
    # 绘制学习曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(train_sizes_abs, train_rmse, 'o-', label='训练集RMSE')
    ax.plot(train_sizes_abs, test_rmse, 'o-', label='验证集RMSE')
    
    ax.set_title('学习曲线')
    ax.set_xlabel('训练样本数')
    ax.set_ylabel('RMSE')
    ax.legend()
    ax.grid(True)
    
    return fig 