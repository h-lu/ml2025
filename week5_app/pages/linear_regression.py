import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import sys
import os

# 添加utils目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.fonts import configure_matplotlib_fonts

def show():
    # 配置matplotlib字体
    configure_matplotlib_fonts()
    
    st.markdown("## 线性回归")
    
    st.markdown("### 线性回归基本原理")
    
    st.markdown("""
    **线性回归**是最简单且应用最广泛的回归算法之一。它假设特征和目标变量之间存在线性关系，即特征的线性组合可以预测目标变量。
    
    **基本模型:**
    
    $$y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + ... + \\beta_n x_n + \\epsilon$$
    
    其中：
    - $y$ 是目标变量
    - $x_1, x_2, ..., x_n$ 是特征变量
    - $\\beta_0, \\beta_1, ..., \\beta_n$ 是模型参数（系数）
    - $\\epsilon$ 是误差项
    
    **矩阵形式:**
    
    $$\\mathbf{y} = \\mathbf{X} \\boldsymbol{\\beta} + \\boldsymbol{\\epsilon}$$
    
    其中：
    - $\\mathbf{y}$ 是目标变量向量 $(m \\times 1)$
    - $\\mathbf{X}$ 是特征矩阵 $(m \\times (n+1))$，包含一列全为1的截距项
    - $\\boldsymbol{\\beta}$ 是参数向量 $((n+1) \\times 1)$
    - $\\boldsymbol{\\epsilon}$ 是误差向量 $(m \\times 1)$
    """)
    
    # 参数估计
    st.markdown("### 参数估计：最小二乘法")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        线性回归通常使用**最小二乘法（Ordinary Least Squares, OLS）**估计参数，其目标是最小化**残差平方和（RSS）**：
        
        $$RSS = \\sum_{i=1}^{m} (y_i - \\hat{y}_i)^2 = \\sum_{i=1}^{m} (y_i - (\\beta_0 + \\beta_1 x_{i1} + ... + \\beta_n x_{in}))^2$$
        
        最小二乘法的**封闭解（closed-form solution）**为：
        
        $$\\boldsymbol{\\beta} = (\\mathbf{X}^T\\mathbf{X})^{-1}\\mathbf{X}^T\\mathbf{y}$$
        
        这个解是唯一的，只要 $\\mathbf{X}^T\\mathbf{X}$ 是可逆的（即特征之间线性独立）。
        """)
    
    with col2:
        # 创建最小二乘法的示意图
        fig, ax = plt.subplots(figsize=(5, 4))
        np.random.seed(42)
        x = np.linspace(0, 10, 20)
        y = 2 * x + 1 + np.random.normal(0, 1.5, 20)
        
        # 拟合线性回归
        lr = LinearRegression()
        lr.fit(x.reshape(-1, 1), y)
        y_pred = lr.predict(x.reshape(-1, 1))
        
        # 绘制数据点和回归线
        ax.scatter(x, y, color='blue', label='数据点')
        ax.plot(x, y_pred, color='red', label='拟合线')
        
        # 绘制残差
        for i in range(len(x)):
            ax.plot([x[i], x[i]], [y[i], y_pred[i]], 'g--', alpha=0.5)
        
        ax.set_xlabel('特征 X')
        ax.set_ylabel('目标 y')
        ax.set_title('线性回归最小二乘法')
        ax.legend()
        st.pyplot(fig)
        st.caption("绿色虚线表示残差，线性回归的目标是使这些残差的平方和最小")
    
    # 假设条件
    st.markdown("### 线性回归的假设条件")
    
    assumptions = {
        "线性关系": "自变量和因变量之间存在线性关系",
        "误差项独立同分布": "观测之间的误差相互独立，且服从相同的分布",
        "误差项均值为零": "误差项的期望为零",
        "同方差性": "误差项的方差在所有自变量取值下保持不变（即误差项的方差不随自变量变化）",
        "无多重共线性": "自变量之间不存在完全线性相关",
        "误差项服从正态分布": "误差项服从正态分布（对大样本的参数推断不是必需的）"
    }
    
    for assumption, description in assumptions.items():
        st.markdown(f"**{assumption}**: {description}")
    
    # 交互式演示
    st.markdown("### 交互式线性回归演示")
    
    st.markdown("#### 生成模拟数据")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_samples = st.slider("样本数量", min_value=10, max_value=200, value=50, step=10)
        noise_level = st.slider("噪声水平", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    
    with col2:
        true_slope = st.slider("真实斜率", min_value=-5.0, max_value=5.0, value=2.0, step=0.1)
        true_intercept = st.slider("真实截距", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)
    
    # 生成数据
    np.random.seed(42)
    X = np.linspace(0, 10, n_samples)
    y = true_intercept + true_slope * X + np.random.normal(0, noise_level, n_samples)
    
    # 拟合线性回归模型
    X_reshaped = X.reshape(-1, 1)
    lr_model = LinearRegression()
    lr_model.fit(X_reshaped, y)
    y_pred = lr_model.predict(X_reshaped)
    
    # 计算评估指标
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # 显示结果
    st.markdown("#### 拟合结果")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("估计斜率", f"{lr_model.coef_[0]:.4f}", f"{lr_model.coef_[0] - true_slope:.4f}")
        st.metric("估计截距", f"{lr_model.intercept_:.4f}", f"{lr_model.intercept_ - true_intercept:.4f}")
    
    with col2:
        st.metric("均方误差 (MSE)", f"{mse:.4f}")
        st.metric("决定系数 (R²)", f"{r2:.4f}")
    
    # 可视化结果
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制数据点和回归线
    ax.scatter(X, y, color='blue', alpha=0.6, label='数据点')
    ax.plot(X, y_pred, color='red', linewidth=2, label=f'拟合线 (y = {lr_model.intercept_:.2f} + {lr_model.coef_[0]:.2f}x)')
    ax.plot(X, true_intercept + true_slope * X, color='green', linestyle='--', 
            label=f'真实线 (y = {true_intercept:.2f} + {true_slope:.2f}x)')
    
    ax.set_xlabel('特征 X')
    ax.set_ylabel('目标 y')
    ax.set_title('线性回归拟合结果')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # 多元线性回归
    st.markdown("### 多元线性回归")
    
    st.markdown("""
    **多元线性回归**是线性回归在多个自变量情况下的扩展。
    
    在多元线性回归中：
    - 模型有多个特征变量 $(x_1, x_2, ..., x_n)$
    - 每个特征有自己的系数 $(\\beta_1, \\beta_2, ..., \\beta_n)$
    - 目标仍然是找到最小化残差平方和的系数
    
    **挑战**:
    - 特征间的相关性（多重共线性）
    - 更多的特征意味着更多的参数需要估计
    - 可能导致过拟合
    
    **解决方案**:
    - 特征选择
    - 正则化（下一课时会详细介绍）
    - 主成分分析等降维技术
    """)
    
    # 线性回归的优缺点
    st.markdown("### 线性回归的优缺点")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 优点")
        st.markdown("""
        - **简单易懂**：模型直观，易于理解和解释
        - **计算效率高**：有封闭解，不需要迭代优化
        - **可解释性强**：系数直接反映特征对目标的影响
        - **扩展性好**：可以通过特征工程捕捉非线性关系
        """)
    
    with col2:
        st.markdown("#### 缺点")
        st.markdown("""
        - **假设限制**：需要满足多个严格假设
        - **对异常值敏感**：离群点会显著影响模型拟合
        - **处理非线性关系能力弱**：原始形式无法捕捉复杂的非线性模式
        - **不适用于高维小样本数据**：容易过拟合
        """)
    
    # 实际应用示例
    st.markdown("### 实际应用案例：波士顿房价预测")
    
    st.markdown("""
    这是一个经典的回归问题示例。虽然原始波士顿房价数据集因为伦理问题已被废弃，我们这里使用类似结构的模拟数据集进行演示。
    """)
    
    # 生成模拟的波士顿房价数据
    np.random.seed(42)
    n_samples = 506  # 原始波士顿数据集的样本量
    n_features = 5
    
    # 模拟特征
    X_boston = np.random.randn(n_samples, n_features)
    feature_names = ["CRIM (犯罪率)", "RM (房间数)", "AGE (房屋年龄)", "DIS (到就业中心距离)", "TAX (税率)"]
    
    # 模拟目标变量（房价）
    coefs = np.array([[-0.5, 3.0, -0.3, -0.7, -0.5]])  # 模拟真实系数
    y_boston = 22.0 + np.dot(X_boston, coefs.T).ravel() + np.random.normal(0, 3, n_samples)
    
    # 创建数据框
    boston_df = pd.DataFrame(X_boston, columns=feature_names)
    boston_df["PRICE"] = y_boston
    
    # 显示数据样本
    st.markdown("#### 数据预览")
    st.dataframe(boston_df.head())
    
    # 特征与目标的关系
    st.markdown("#### 各特征与房价的关系")
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, feature in enumerate(feature_names):
        axes[i].scatter(boston_df[feature], boston_df["PRICE"], alpha=0.5)
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("PRICE")
        axes[i].set_title(f"{feature} vs PRICE")
    plt.tight_layout()
    st.pyplot(fig)
    
    # 模型训练
    st.markdown("#### 模型训练")
    
    X_train, X_test, y_train, y_test = train_test_split(
        boston_df.drop("PRICE", axis=1), boston_df["PRICE"], test_size=0.2, random_state=42
    )
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练模型
    boston_model = LinearRegression()
    boston_model.fit(X_train_scaled, y_train)
    
    # 预测
    y_train_pred = boston_model.predict(X_train_scaled)
    y_test_pred = boston_model.predict(X_test_scaled)
    
    # 评估指标
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # 显示评估结果
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("训练集 MSE", f"{train_mse:.4f}")
        st.metric("测试集 MSE", f"{test_mse:.4f}")
    
    with col2:
        st.metric("训练集 R²", f"{train_r2:.4f}")
        st.metric("测试集 R²", f"{test_r2:.4f}")
    
    # 显示模型系数
    st.markdown("#### 模型系数")
    
    coef_df = pd.DataFrame({
        "特征": feature_names,
        "系数": boston_model.coef_,
        "绝对值": np.abs(boston_model.coef_)
    })
    coef_df = coef_df.sort_values("绝对值", ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(coef_df["特征"], coef_df["系数"])
    ax.set_xlabel("系数值")
    ax.set_title("特征系数及重要性")
    ax.grid(axis="x", alpha=0.3)
    st.pyplot(fig)
    
    st.markdown("""
    从系数可以看出：
    - **RM (房间数)** 对房价有较强的正面影响
    - **DIS (到就业中心距离)** 对房价有负面影响
    - **CRIM (犯罪率)** 和 **TAX (税率)** 也对房价有负面影响
    - **AGE (房屋年龄)** 对房价影响相对较小
    
    这些系数的含义与我们的常识相符：房间数多的房子价格更高，而距离就业中心远、犯罪率高或税率高的地区房价往往更低。
    """)
    
    # scikit-learn实现
    st.markdown("### scikit-learn线性回归实现")
    
    st.code("""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 标准化特征（可选，但通常有助于提高模型性能）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 创建并训练模型
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # 模型参数
    print("系数:", model.coef_)
    print("截距:", model.intercept_)
    
    # 预测
    y_pred = model.predict(X_test_scaled)
    
    # 评估模型
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("MSE:", mse)
    print("R²:", r2)
    """, language="python")

if __name__ == "__main__":
    show() 