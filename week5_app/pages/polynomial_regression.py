import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, learning_curve
import sys
import os

# 添加utils目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.fonts import configure_matplotlib_fonts

def show():
    # 配置matplotlib字体
    configure_matplotlib_fonts()
    
    st.markdown("## 多项式回归")
    
    st.markdown("### 多项式回归基本原理")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        **多项式回归**是线性回归的扩展，通过引入高阶特征捕捉非线性关系。虽然名称中有"多项式"，但本质上它仍然是一种线性模型（线性是指参数的线性组合，而非特征的线性组合）。
        
        **基本模型:**
        
        $$y = \\beta_0 + \\beta_1 x + \\beta_2 x^2 + ... + \\beta_n x^n + \\epsilon$$
        
        其中，$x, x^2, ..., x^n$ 是原始特征 $x$ 的多项式特征，$\\beta_0, \\beta_1, ..., \\beta_n$ 是模型参数。
        
        **对于多变量情况**，多项式特征包括原始特征的高阶项及它们的交叉项，例如对于两个特征 $x_1$ 和 $x_2$，二阶多项式特征包括:
        $1, x_1, x_2, x_1^2, x_1x_2, x_2^2$。
        """)
    
    with col2:
        # 创建多项式回归的示例图
        np.random.seed(0)
        x = np.linspace(-3, 3, 100).reshape(-1, 1)
        y = 1 + 2*x - 1.5*x**2 + 0.5*x**3 + np.random.randn(100, 1)*1.5
        
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(x, y, color='blue', alpha=0.6, label='数据点')
        
        for degree in [1, 3, 5]:
            poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            x_poly = poly_features.fit_transform(x)
            lin_reg = LinearRegression()
            lin_reg.fit(x_poly, y)
            y_pred = lin_reg.predict(x_poly)
            
            label = f"{degree}阶多项式"
            if degree == 1:
                label = "线性"
            ax.plot(x, y_pred, label=label, alpha=0.7)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('多项式回归与线性回归对比')
        ax.legend()
        st.pyplot(fig)
    
    st.markdown("### 多项式特征转换")
    
    st.markdown("""
    多项式回归的关键在于**特征转换**，通过将原始特征转换为多项式特征，然后应用线性回归方法。
    
    以单变量情况为例，对于原始特征 $x$，其 $n$ 阶多项式特征为：
    
    $$[x, x^2, x^3, ..., x^n]$$
    
    对于两个特征 $x_1$ 和 $x_2$，其二阶多项式特征为：
    
    $$[x_1, x_2, x_1^2, x_1x_2, x_2^2]$$
    
    在 `scikit-learn` 中，我们可以使用 `PolynomialFeatures` 类来生成这些特征。
    """)
    
    # 展示多项式特征转换的效果
    st.markdown("#### 多项式特征转换示例")
    
    n_features_slider = st.slider("特征数量", min_value=1, max_value=3, value=2)
    degree_slider = st.slider("多项式阶数", min_value=1, max_value=5, value=2)
    
    # 生成原始特征矩阵
    np.random.seed(0)
    X_orig = np.random.rand(5, n_features_slider)
    col_names = [f"x{i+1}" for i in range(n_features_slider)]
    
    # 显示原始特征矩阵
    st.markdown("**原始特征:**")
    st.dataframe(pd.DataFrame(X_orig, columns=col_names))
    
    # 转换为多项式特征
    poly_features = PolynomialFeatures(degree=degree_slider, include_bias=True)
    X_poly = poly_features.fit_transform(X_orig)
    
    # 获取特征名称
    feature_names = poly_features.get_feature_names_out(col_names)
    
    # 显示多项式特征矩阵
    st.markdown(f"**{degree_slider}阶多项式特征:**")
    st.dataframe(pd.DataFrame(X_poly, columns=feature_names))
    
    st.markdown(f"原始特征维度: {X_orig.shape[1]} → 多项式特征维度: {X_poly.shape[1]}")
    
    # 交互式多项式回归演示
    st.markdown("### 交互式多项式回归演示")
    
    st.markdown("#### 数据生成与模型拟合")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        data_type = st.selectbox(
            "数据类型",
            ["线性", "二次曲线", "正弦曲线", "指数曲线"]
        )
    
    with col2:
        noise_level = st.slider("噪声水平", min_value=0.0, max_value=2.0, value=0.5, step=0.1)
    
    with col3:
        max_degree = st.slider("最大多项式阶数", min_value=1, max_value=10, value=5)
    
    # 生成数据
    np.random.seed(42)
    x = np.linspace(-3, 3, 100).reshape(-1, 1)
    
    if data_type == "线性":
        y_true = 1.5 * x + 2
        y = y_true + np.random.normal(0, noise_level, (100, 1))
    elif data_type == "二次曲线":
        y_true = 1.5 * x**2 + 2 * x + 1
        y = y_true + np.random.normal(0, noise_level, (100, 1))
    elif data_type == "正弦曲线":
        y_true = np.sin(2 * x) + 0.5 * x
        y = y_true + np.random.normal(0, noise_level, (100, 1))
    else:  # 指数曲线
        y_true = np.exp(0.5 * x) / 5
        y = y_true + np.random.normal(0, noise_level, (100, 1))
    
    # 拟合多项式回归模型
    models = {}
    mse_scores = {}
    r2_scores = {}
    
    for degree in range(1, max_degree + 1):
        model = Pipeline([
            ("poly_features", PolynomialFeatures(degree=degree, include_bias=True)),
            ("lin_reg", LinearRegression())
        ])
        
        model.fit(x, y)
        y_pred = model.predict(x)
        
        models[degree] = model
        mse_scores[degree] = mean_squared_error(y, y_pred)
        r2_scores[degree] = r2_score(y, y_pred)
    
    # 可视化结果
    st.markdown("#### 拟合结果可视化")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制原始数据
    ax.scatter(x, y, color='blue', alpha=0.5, label='数据点')
    ax.plot(x, y_true, color='green', linestyle='--', label='真实函数')
    
    # 绘制各阶多项式拟合结果
    degree_to_show = st.multiselect(
        "选择要显示的多项式阶数",
        list(range(1, max_degree + 1)),
        default=[1, min(3, max_degree), max_degree]
    )
    
    for degree in degree_to_show:
        y_pred = models[degree].predict(x)
        ax.plot(x, y_pred, label=f"{degree}阶多项式")
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('多项式回归拟合结果')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # 显示评估指标
    st.markdown("#### 各阶多项式的性能评估")
    
    metrics_df = pd.DataFrame({
        "多项式阶数": list(range(1, max_degree + 1)),
        "MSE": [mse_scores[i] for i in range(1, max_degree + 1)],
        "R²": [r2_scores[i] for i in range(1, max_degree + 1)]
    })
    
    st.dataframe(metrics_df)
    
    # 绘制MSE和R²随阶数的变化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(range(1, max_degree + 1), [mse_scores[i] for i in range(1, max_degree + 1)], 'o-')
    ax1.set_xlabel('多项式阶数')
    ax1.set_ylabel('MSE')
    ax1.set_title('MSE vs 多项式阶数')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(range(1, max_degree + 1), [r2_scores[i] for i in range(1, max_degree + 1)], 'o-')
    ax2.set_xlabel('多项式阶数')
    ax2.set_ylabel('R²')
    ax2.set_title('R² vs 多项式阶数')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 过拟合与欠拟合
    st.markdown("### 过拟合与欠拟合")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 欠拟合 (Underfitting)")
        st.markdown("""
        **特点**:
        - 模型过于简单，无法捕捉数据的复杂模式
        - 高偏差 (high bias)
        - 训练误差和测试误差都很高
        - 典型例子：对非线性数据使用线性模型
        
        **解决方法**:
        - 增加模型复杂度（如提高多项式阶数）
        - 添加更多有意义的特征
        - 减少正则化强度
        """)
    
    with col2:
        st.markdown("#### 过拟合 (Overfitting)")
        st.markdown("""
        **特点**:
        - 模型过于复杂，学习了数据中的噪声
        - 高方差 (high variance)
        - 训练误差极低，但测试误差很高
        - 典型例子：对简单数据使用高阶多项式
        
        **解决方法**:
        - 降低模型复杂度（如降低多项式阶数）
        - 增加训练数据量
        - 应用正则化技术
        - 使用交叉验证选择合适的模型
        """)
    
    # 交互式演示欠拟合和过拟合
    st.markdown("#### 欠拟合与过拟合的交互式演示")
    
    col1, col2 = st.columns(2)
    
    with col1:
        train_size = st.slider("训练集样本数量", min_value=10, max_value=80, value=30, step=5)
    
    with col2:
        fit_degree = st.slider("拟合多项式阶数", min_value=1, max_value=15, value=5, step=1)
    
    # 生成数据
    np.random.seed(0)
    x_full = np.linspace(-3, 3, 100).reshape(-1, 1)
    x_train = np.random.uniform(-3, 3, train_size).reshape(-1, 1)
    y_true_func = lambda x: np.sin(1.5 * x) * x + 0.1 * x**2
    y_true_full = y_true_func(x_full)
    y_true_train = y_true_func(x_train)
    y_train = y_true_train + np.random.normal(0, 0.2, (train_size, 1))
    
    # 拟合模型
    model = Pipeline([
        ("poly_features", PolynomialFeatures(degree=fit_degree, include_bias=True)),
        ("lin_reg", LinearRegression())
    ])
    
    model.fit(x_train, y_train)
    y_pred_full = model.predict(x_full)
    y_pred_train = model.predict(x_train)
    
    # 计算训练误差
    train_mse = mean_squared_error(y_train, y_pred_train)
    
    # 绘制结果
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制真实函数
    ax.plot(x_full, y_true_full, color='green', linestyle='--', label='真实函数')
    
    # 绘制训练数据
    ax.scatter(x_train, y_train, color='blue', alpha=0.6, label='训练数据')
    
    # 绘制拟合结果
    ax.plot(x_full, y_pred_full, color='red', label=f"{fit_degree}阶多项式拟合")
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'多项式回归：训练集大小={train_size}，阶数={fit_degree}，MSE={train_mse:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    st.markdown("""
    **观察**:
    - 当阶数过低（如1阶）时，模型无法捕捉数据的非线性特征，出现**欠拟合**
    - 当阶数过高（如15阶）时，模型开始拟合噪声，在数据点之间出现大幅振荡，这是**过拟合**的特征
    - 适当的阶数能够较好地近似真实函数，既不会过于简单，也不会捕捉太多噪声
    
    **调整样本量的影响**:
    - 样本量大时，即使是高阶多项式也不容易过拟合
    - 样本量小时，高阶多项式很容易过拟合
    
    这说明过拟合与欠拟合不仅与模型复杂度有关，还与数据量有密切关系。
    """)
    
    # 学习曲线
    st.markdown("#### 学习曲线分析")
    
    st.markdown("""
    **学习曲线**是评估模型性能的重要工具，它展示了模型在不同训练样本量下的训练误差和验证误差。
    
    从学习曲线可以判断模型是欠拟合还是过拟合：
    - 如果训练误差和验证误差都很高，且相近，则模型可能欠拟合
    - 如果训练误差低但验证误差高，则模型可能过拟合
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        degree1 = st.selectbox("低阶多项式", [1, 2, 3], index=0)
    with col2:
        degree2 = st.selectbox("中阶多项式", [4, 5, 6], index=1)
    with col3:
        degree3 = st.selectbox("高阶多项式", [10, 12, 15], index=2)
    
    learning_degree = [degree1, degree2, degree3]
    
    # 生成更多数据用于学习曲线分析
    np.random.seed(0)
    X_learn = np.random.uniform(-3, 3, 200).reshape(-1, 1)
    y_learn = y_true_func(X_learn) + np.random.normal(0, 0.2, (200, 1))
    
    # 绘制学习曲线
    fig, axes = plt.subplots(1, len(learning_degree), figsize=(15, 5))
    
    for i, degree in enumerate(learning_degree):
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_learn)
        
        train_sizes, train_scores, valid_scores = learning_curve(
            LinearRegression(), X_poly, y_learn, train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5, scoring='neg_mean_squared_error'
        )
        
        train_scores_mean = -np.mean(train_scores, axis=1)
        valid_scores_mean = -np.mean(valid_scores, axis=1)
        
        axes[i].plot(train_sizes, train_scores_mean, 'o-', color='r', label='训练误差')
        axes[i].plot(train_sizes, valid_scores_mean, 'o-', color='g', label='验证误差')
        axes[i].set_title(f"{degree}阶多项式")
        axes[i].set_xlabel('训练样本数')
        axes[i].set_ylabel('MSE')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    从学习曲线可以观察到：
    - **低阶多项式**：训练误差和验证误差都较高，且相近，表明模型欠拟合
    - **高阶多项式**：训练误差很低，但验证误差高，且两者差距大，表明模型过拟合
    - **适中阶数**：训练误差和验证误差都降低，且差距适中，表明模型拟合较好
    
    随着训练样本数的增加，验证误差通常会降低，尤其是对于高阶多项式，这表明增加数据量可以减轻过拟合问题。
    """)
    
    # scikit-learn实现
    st.markdown("### scikit-learn实现多项式回归")
    
    st.code("""
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    
    # 生成或加载数据
    X = ...  # 特征
    y = ...  # 目标
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 创建多项式回归模型
    degree = 3  # 多项式阶数
    poly_reg = Pipeline([
        ("poly_features", PolynomialFeatures(degree=degree, include_bias=True)),
        ("lin_reg", LinearRegression())
    ])
    
    # 训练模型
    poly_reg.fit(X_train, y_train)
    
    # 预测
    y_train_pred = poly_reg.predict(X_train)
    y_test_pred = poly_reg.predict(X_test)
    
    # 评估模型
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("训练集 MSE:", train_mse)
    print("测试集 MSE:", test_mse)
    print("训练集 R²:", train_r2)
    print("测试集 R²:", test_r2)
    """, language="python")
    
    # 小结
    st.markdown("### 小结")
    
    st.markdown("""
    **多项式回归**是一种通过引入高阶特征捕捉非线性关系的方法。尽管它比简单的线性回归更灵活，但也面临过拟合的风险，尤其是在数据量小而多项式阶数高的情况下。
    
    **主要优点**:
    - 能够捕捉数据中的非线性关系
    - 模型仍然具有线性回归的简单性和可解释性
    - 使用标准的线性回归技术进行参数估计
    
    **主要挑战**:
    - 高阶多项式容易过拟合
    - 特征数量随多项式阶数指数增长
    - 需要谨慎选择合适的阶数
    
    **应对过拟合的方法**:
    - 正则化（如Ridge, Lasso）
    - 交叉验证选择合适的多项式阶数
    - 增加训练数据量
    
    多项式回归是连接简单线性模型和更复杂非线性模型之间的重要桥梁，掌握它有助于更好地理解机器学习中的复杂性-泛化能力权衡。
    """)

if __name__ == "__main__":
    show() 