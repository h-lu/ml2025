import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import sys
import os

# 添加utils目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.fonts import configure_matplotlib_fonts

def show():
    # 配置matplotlib字体
    configure_matplotlib_fonts()
    
    st.markdown("## 回归模型评估")
    
    st.markdown("### 回归评估指标")
    
    metrics = {
        "均方误差 (MSE)": {
            "公式": r"$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2$",
            "特点": "对异常值敏感，越小越好",
            "取值范围": "[0, +∞)",
            "适用场景": "当异常值的惩罚需要更高时使用"
        },
        "均方根误差 (RMSE)": {
            "公式": r"$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}$",
            "特点": "与MSE相同，但单位与目标变量相同",
            "取值范围": "[0, +∞)",
            "适用场景": "需要与目标变量相同单位的误差度量"
        },
        "平均绝对误差 (MAE)": {
            "公式": r"$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|$",
            "特点": "对异常值不敏感，越小越好",
            "取值范围": "[0, +∞)",
            "适用场景": "当异常值的影响需要减小时使用"
        },
        "决定系数 (R²)": {
            "公式": r"$\text{R}^2 = 1 - \frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{n}(y_i-\bar{y})^2}$",
            "特点": "表示模型解释的目标变量方差比例",
            "取值范围": "(-∞, 1]，1表示完美拟合",
            "适用场景": "比较不同模型的表现，理解模型的解释能力"
        },
        "调整R² (Adjusted R²)": {
            "公式": r"$\text{Adj. R}^2 = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$",
            "特点": "考虑特征数量的R²修正版",
            "取值范围": "(-∞, 1]",
            "适用场景": "比较具有不同特征数量的模型"
        }
    }
    
    # 创建评估指标表格
    metrics_df = pd.DataFrame(metrics).T
    st.table(metrics_df)
    
    # 交互式演示评估指标
    st.markdown("### 交互式评估指标演示")
    
    st.markdown("""
    在这个演示中，我们将生成一些带有不同程度噪声的数据，并计算不同回归评估指标的值。
    调整滑块来改变数据的特性，看看不同指标是如何变化的。
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        noise_level = st.slider("噪声水平", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
        outlier_strength = st.slider("异常值强度", min_value=0.0, max_value=20.0, value=0.0, step=0.5)
    
    with col2:
        sample_size = st.slider("样本大小", min_value=20, max_value=200, value=50, step=10)
        model_complexity = st.slider("模型复杂度", min_value=1, max_value=10, value=1, step=1)
    
    # 生成数据
    np.random.seed(42)
    X = np.linspace(0, 10, sample_size).reshape(-1, 1)
    y_true = 3 * X.ravel() + 2
    y = y_true + np.random.normal(0, noise_level, sample_size)
    
    # 添加异常值
    if outlier_strength > 0:
        outlier_idx = np.random.choice(sample_size, size=int(sample_size * 0.05), replace=False)
        y[outlier_idx] += np.random.choice([-1, 1], size=len(outlier_idx)) * outlier_strength
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 创建并训练模型
    if model_complexity == 1:
        model = LinearRegression()
        model.fit(X_train, y_train)
        label = "线性回归"
    else:
        model = Pipeline([
            ("poly", PolynomialFeatures(degree=model_complexity, include_bias=True)),
            ("linear", LinearRegression())
        ])
        model.fit(X_train, y_train)
        label = f"{model_complexity}阶多项式回归"
    
    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 计算评估指标
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # 计算调整R²
    n_train = len(y_train)
    p = model_complexity  # 特征数量/模型复杂度
    train_adj_r2 = 1 - (1 - train_r2) * (n_train - 1) / (n_train - p - 1)
    
    n_test = len(y_test)
    test_adj_r2 = 1 - (1 - test_r2) * (n_test - 1) / (n_test - p - 1)
    
    # 显示评估结果
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 训练集评估指标")
        st.metric("MSE", f"{train_mse:.4f}")
        st.metric("RMSE", f"{train_rmse:.4f}")
        st.metric("MAE", f"{train_mae:.4f}")
        st.metric("R²", f"{train_r2:.4f}")
        st.metric("调整R²", f"{train_adj_r2:.4f}")
    
    with col2:
        st.markdown("#### 测试集评估指标")
        st.metric("MSE", f"{test_mse:.4f}")
        st.metric("RMSE", f"{test_rmse:.4f}")
        st.metric("MAE", f"{test_mae:.4f}")
        st.metric("R²", f"{test_r2:.4f}")
        st.metric("调整R²", f"{test_adj_r2:.4f}")
    
    # 可视化结果
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(X_train, y_train, color='blue', alpha=0.6, label='训练数据')
    ax.scatter(X_test, y_test, color='green', alpha=0.6, label='测试数据')
    
    # 绘制预测线
    X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    ax.plot(X_plot, y_plot, color='red', label=label)
    
    ax.set_xlabel('特征 X')
    ax.set_ylabel('目标 y')
    ax.set_title(f'模型拟合结果（测试集R²={test_r2:.4f}）')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    st.markdown("""
    **观察**:
    - 增加**噪声水平**会增加所有误差指标，降低R²值
    - 添加**异常值**对MSE和RMSE的影响比对MAE的影响大
    - 增加**模型复杂度**在简单数据上可能导致过拟合，表现为训练集指标优于测试集指标
    - **调整R²**会惩罚不必要的复杂模型，当添加的特征没有提供额外信息时，它会降低
    """)
    
    # 正则化
    st.markdown("### 正则化技术")
    
    st.markdown("""
    **正则化**是一种防止过拟合的重要技术，特别是在特征数量多或模型复杂度高的情况下。正则化通过向损失函数添加惩罚项，限制模型参数的大小，从而抑制模型的复杂度。
    
    主要的正则化技术包括：
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### L1正则化 (LASSO)")
        st.markdown("""
        **损失函数**:
        
        $$J(\\theta) = MSE(\\theta) + \\alpha \\sum_{i=1}^{n} |\\theta_i|$$
        
        **特点**:
        - 可以将某些系数压缩为零，进行特征选择
        - 产生稀疏模型
        - 适用于特征很多但只有少数重要特征的情况
        
        **scikit-learn实现**: `Lasso`
        """)
    
    with col2:
        st.markdown("#### L2正则化 (Ridge)")
        st.markdown("""
        **损失函数**:
        
        $$J(\\theta) = MSE(\\theta) + \\alpha \\sum_{i=1}^{n} \\theta_i^2$$
        
        **特点**:
        - 压缩所有系数，但通常不会将系数压缩为零
        - 有助于处理多重共线性问题
        - 通常比L1正则化结果更稳定
        
        **scikit-learn实现**: `Ridge`
        """)
    
    st.markdown("#### 弹性网络 (Elastic Net)")
    st.markdown("""
    **损失函数**:
    
    $$J(\\theta) = MSE(\\theta) + r\\alpha \\sum_{i=1}^{n} |\\theta_i| + \\frac{(1-r)\\alpha}{2} \\sum_{i=1}^{n} \\theta_i^2$$
    
    **特点**:
    - 结合了L1和L2正则化的优点
    - $r$ 控制L1和L2正则化的混合比例
    - 既能进行特征选择，又能处理多重共线性
    
    **scikit-learn实现**: `ElasticNet`
    """)
    
    # 交互式演示正则化效果
    st.markdown("### 交互式正则化演示")
    
    st.markdown("""
    在这个演示中，我们将比较无正则化、L1正则化（LASSO）和L2正则化（Ridge）的效果。
    调整滑块来改变数据的特性和正则化强度，观察结果的变化。
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        noise_level_reg = st.slider("噪声水平", min_value=0.0, max_value=5.0, value=2.0, step=0.1, key="noise_reg")
        sample_size_reg = st.slider("样本大小", min_value=20, max_value=200, value=50, step=10, key="sample_reg")
    
    with col2:
        n_features = st.slider("特征数量", min_value=1, max_value=20, value=10, step=1)
        alpha = st.slider("正则化强度 (α)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    # 生成数据
    np.random.seed(42)
    X_reg = np.random.randn(sample_size_reg, n_features)
    
    # 只有前3个特征是真正有用的
    true_coef = np.zeros(n_features)
    true_coef[:3] = [3, 1.5, -2]
    
    y_reg_true = np.dot(X_reg, true_coef)
    y_reg = y_reg_true + np.random.normal(0, noise_level_reg, size=sample_size_reg)
    
    # 划分训练集和测试集
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    # 训练不同的模型
    lr = LinearRegression()
    lr.fit(X_reg_train, y_reg_train)
    
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_reg_train, y_reg_train)
    
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_reg_train, y_reg_train)
    
    # 预测
    y_reg_pred_lr = lr.predict(X_reg_test)
    y_reg_pred_ridge = ridge.predict(X_reg_test)
    y_reg_pred_lasso = lasso.predict(X_reg_test)
    
    # 计算评估指标
    lr_mse = mean_squared_error(y_reg_test, y_reg_pred_lr)
    ridge_mse = mean_squared_error(y_reg_test, y_reg_pred_ridge)
    lasso_mse = mean_squared_error(y_reg_test, y_reg_pred_lasso)
    
    lr_r2 = r2_score(y_reg_test, y_reg_pred_lr)
    ridge_r2 = r2_score(y_reg_test, y_reg_pred_ridge)
    lasso_r2 = r2_score(y_reg_test, y_reg_pred_lasso)
    
    # 显示评估结果
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 无正则化")
        st.metric("MSE", f"{lr_mse:.4f}")
        st.metric("R²", f"{lr_r2:.4f}")
    
    with col2:
        st.markdown("#### Ridge (L2)")
        st.metric("MSE", f"{ridge_mse:.4f}")
        st.metric("R²", f"{ridge_r2:.4f}")
    
    with col3:
        st.markdown("#### LASSO (L1)")
        st.metric("MSE", f"{lasso_mse:.4f}")
        st.metric("R²", f"{lasso_r2:.4f}")
    
    # 绘制系数对比
    fig, ax = plt.subplots(figsize=(12, 6))
    
    feature_names = [f"特征 {i+1}" for i in range(n_features)]
    x = np.arange(len(feature_names))
    width = 0.25
    
    ax.bar(x - width, true_coef, width, label='真实系数')
    ax.bar(x, lr.coef_, width, label='无正则化')
    ax.bar(x + width, ridge.coef_, width, label='Ridge (L2)')
    ax.bar(x + 2*width, lasso.coef_, width, label='LASSO (L1)')
    
    ax.set_xlabel('特征')
    ax.set_ylabel('系数')
    ax.set_title('各类模型系数对比')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    **观察**:
    - **无正则化**的线性回归可能会过拟合，特别是在特征多但样本少的情况下，导致系数不稳定
    - **Ridge回归**会收缩所有系数，降低模型复杂度，但通常不会完全消除某些特征的影响
    - **LASSO回归**可以将不重要特征的系数压缩为零，实现特征选择
    
    **应用建议**:
    - 当特征间存在多重共线性时，考虑使用**Ridge回归**
    - 当需要获得稀疏模型或进行特征选择时，考虑使用**LASSO回归**
    - 正则化强度 $\\alpha$ 是一个重要的超参数，通常通过交叉验证选择
    """)
    
    # 交叉验证
    st.markdown("### 交叉验证")
    
    st.markdown("""
    **交叉验证**是一种评估模型性能的重要技术，特别是在数据量有限的情况下。它通过将数据多次分割成训练集和验证集，并在每次分割上训练和评估模型，从而获得更稳定可靠的性能估计。
    
    **常见的交叉验证方法**:
    
    1. **K折交叉验证 (K-Fold CV)**：
       - 将数据集分成K个相等的子集（折）
       - 每次使用K-1个折进行训练，1个折进行验证
       - 重复K次，每次使用不同的折作为验证集
       - 最终结果是K次验证的平均值
    
    2. **留一交叉验证 (Leave-One-Out CV)**：
       - K折交叉验证的极端情况，K等于样本数
       - 每次只用一个样本进行验证
       - 计算量大，但对小数据集很有用
    
    3. **分层K折交叉验证**：
       - 在分类问题中保持各折中类别分布一致
    """)
    
    # 交互式演示交叉验证
    st.markdown("### 交互式交叉验证演示")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_folds = st.slider("K折数", min_value=2, max_value=10, value=5, step=1)
        cv_noise = st.slider("噪声水平", min_value=0.0, max_value=3.0, value=1.0, step=0.1, key="cv_noise")
    
    with col2:
        cv_degree = st.multiselect("多项式阶数", options=[1, 3, 5, 7, 9], default=[1, 5, 9])
        if not cv_degree:  # 如果用户没有选择任何值，使用默认值
            cv_degree = [1, 5, 9]
    
    # 生成数据
    np.random.seed(42)
    X_cv = np.sort(np.random.rand(50, 1) * 10, axis=0)
    y_cv = np.sin(X_cv).ravel() + np.random.normal(0, cv_noise, 50)
    
    # 创建KFold对象
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # 计算每个多项式阶数的交叉验证分数
    degrees = sorted(cv_degree)
    cv_scores = []
    
    for degree in degrees:
        model = Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
            ("linear", LinearRegression())
        ])
        scores = cross_val_score(model, X_cv, y_cv, cv=kf, scoring='neg_mean_squared_error')
        cv_scores.append(-scores)  # 转换为正MSE
    
    # 计算每个折的训练/验证索引
    train_indices = []
    val_indices = []
    
    for train_idx, val_idx in kf.split(X_cv):
        train_indices.append(train_idx)
        val_indices.append(val_idx)
    
    # 绘制交叉验证示意图
    st.markdown(f"#### {n_folds}折交叉验证示意图")
    
    fig, axes = plt.subplots(n_folds, 1, figsize=(10, 2*n_folds))
    if n_folds == 1:
        axes = [axes]
    
    for i, (train_idx, val_idx) in enumerate(zip(train_indices, val_indices)):
        axes[i].scatter(range(len(X_cv)), np.zeros_like(X_cv), c=['blue' if j in train_idx else 'red' for j in range(len(X_cv))])
        axes[i].set_yticks([])
        axes[i].set_title(f"折 {i+1}")
        axes[i].set_xlabel("样本" if i == n_folds-1 else "")
        axes[i].legend(['训练集', '验证集'], loc='upper right')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 绘制不同多项式阶数的交叉验证MSE
    st.markdown("#### 不同多项式阶数的交叉验证MSE")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, degree in enumerate(degrees):
        ax.boxplot(cv_scores[i], positions=[degree], widths=0.5)
    
    # 计算每个阶数的平均MSE
    mean_scores = [np.mean(scores) for scores in cv_scores]
    ax.plot(degrees, mean_scores, 'r-', label='平均MSE')
    
    ax.set_xlabel('多项式阶数')
    ax.set_ylabel('MSE')
    ax.set_title('交叉验证：不同多项式阶数的MSE分布')
    ax.set_xticks(degrees)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # 找出最佳阶数
    best_degree_idx = np.argmin(mean_scores)
    best_degree = degrees[best_degree_idx]
    
    st.markdown(f"**最佳多项式阶数**: {best_degree} (平均MSE: {mean_scores[best_degree_idx]:.4f})")
    
    # 拟合最佳模型并可视化
    best_model = Pipeline([
        ("poly", PolynomialFeatures(degree=best_degree, include_bias=True)),
        ("linear", LinearRegression())
    ])
    
    best_model.fit(X_cv, y_cv)
    
    X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
    y_plot = best_model.predict(X_plot)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(X_cv, y_cv, color='blue', alpha=0.6, label='原始数据')
    ax.plot(X_plot, y_plot, color='red', label=f'{best_degree}阶多项式（最佳）')
    
    # 也绘制其他阶数的拟合结果
    for degree in degrees:
        if degree != best_degree:
            model = Pipeline([
                ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
                ("linear", LinearRegression())
            ])
            model.fit(X_cv, y_cv)
            y_plot = model.predict(X_plot)
            ax.plot(X_plot, y_plot, '--', alpha=0.5, label=f'{degree}阶多项式')
    
    ax.set_xlabel('特征 X')
    ax.set_ylabel('目标 y')
    ax.set_title('交叉验证选择的最佳模型')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    st.markdown("""
    **交叉验证的重要性**:
    
    - 提供对模型泛化能力的更可靠估计
    - 减少单次训练-测试拆分带来的随机性
    - 帮助选择最优超参数（如多项式阶数、正则化强度等）
    - 特别适用于数据量有限的情况
    
    **使用交叉验证的建议**:
    
    - 一般情况下，5折或10折交叉验证是常用选择
    - 在数据量非常小时，可以考虑留一交叉验证
    - 对于不平衡数据集，应使用分层交叉验证
    - 交叉验证虽然计算量大，但提供的性能估计更加可靠
    """)
    
    # scikit-learn实现
    st.markdown("### scikit-learn实现交叉验证和正则化")
    
    st.code("""
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import mean_squared_error, r2_score
    
    # 加载数据
    X = ...  # 特征
    y = ...  # 目标
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 创建模型管道（包含标准化、多项式特征和正则化回归）
    ridge_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(include_bias=False)),
        ('ridge', Ridge())
    ])
    
    # 创建超参数网格
    param_grid = {
        'poly__degree': [1, 2, 3, 4],
        'ridge__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    }
    
    # 使用网格搜索和交叉验证找到最佳超参数
    grid_search = GridSearchCV(
        ridge_pipeline, param_grid, cv=5, 
        scoring='neg_mean_squared_error', 
        return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    
    # 输出最佳参数和得分
    print("最佳参数:", grid_search.best_params_)
    print("最佳交叉验证得分:", -grid_search.best_score_)
    
    # 在测试集上评估最佳模型
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("测试集MSE:", mean_squared_error(y_test, y_pred))
    print("测试集R²:", r2_score(y_test, y_pred))
    """, language="python")
    
    # 小结
    st.markdown("### 小结")
    
    st.markdown("""
    **回归模型评估**是建立可靠预测模型的关键步骤。通过选择合适的评估指标、应用正则化技术和使用交叉验证，可以有效提高模型的泛化能力。
    
    **关键要点**:
    
    1. **评估指标**:
       - MSE/RMSE适用于异常值惩罚需要较高的情况
       - MAE对异常值不敏感，提供更稳健的度量
       - R²提供模型解释能力的度量，但在某些情况下可能产生误导
    
    2. **正则化**:
       - L1正则化(LASSO)有助于特征选择，产生稀疏模型
       - L2正则化(Ridge)适用于处理多重共线性问题
       - 正则化强度是需要调优的超参数
    
    3. **交叉验证**:
       - 提供对模型泛化能力的可靠估计
       - 帮助选择最优模型和超参数
       - 减少单次训练-测试拆分的随机性
    
    在实际应用中，应结合问题特点选择合适的评估策略，并通过多种指标综合评估模型性能。
    """)

if __name__ == "__main__":
    show()