import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.svg_generator import render_svg
from utils.matplotlib_charts import create_sigmoid_chart, create_svm_concept_chart, create_model_comparison_chart

def show_concept_explorer():
    """概念探索主功能"""
    st.header("概念探索")
    
    st.markdown("""
    这个工具可以帮助你直观理解逻辑回归和支持向量机(SVM)的核心概念。
    通过交互式可视化，你可以探索这些算法的工作原理和关键特性。
    """)
    
    # 创建选项卡
    tab1, tab2, tab3 = st.tabs(["逻辑回归概念", "SVM概念", "模型比较"])
    
    with tab1:
        show_logistic_regression_concept()
    
    with tab2:
        show_svm_concept()
    
    with tab3:
        show_model_comparison()

def show_logistic_regression_concept():
    """展示逻辑回归的核心概念"""
    st.subheader("逻辑回归核心概念")
    
    st.markdown("""
    逻辑回归是一种基本的分类算法，它使用Sigmoid函数将线性模型的输出转换为概率值。
    Sigmoid函数的输出范围在0到1之间，我们通常将输出大于0.5的样本分类为正类，否则为负类。
    """)
    
    # 显示Sigmoid函数图 - 使用matplotlib代替SVG
    sigmoid_fig = create_sigmoid_chart()
    st.pyplot(sigmoid_fig)
    
    st.markdown("""
    **逻辑回归的关键概念**:
    
    1. **Sigmoid函数**: 将线性函数 $z = w^T x + b$ 的值映射到概率空间 [0,1]
    
    2. **决策边界**: 当 $P(y=1|x) = 0.5$ 时，对应的 $z = 0$，这是分隔两个类别的边界
    
    3. **概率解释**: 输出可以解释为样本属于正类的概率，使得模型更具可解释性
    """)
    
    # 展示逻辑回归决策过程
    st.markdown("### 逻辑回归决策过程")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        1. **线性组合**: 首先计算特征的线性组合
           - z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ = W·X + b
        
        2. **概率转换**: 通过Sigmoid函数将线性组合转换为概率
           - P(y=1|X) = σ(z) = 1/(1+e^(-z))
        
        3. **决策规则**: 如果概率大于阈值(通常为0.5)，预测为正类，否则为负类
           - 当P(y=1|X) > 0.5时，预测y=1
           - 当P(y=1|X) ≤ 0.5时，预测y=0
        """)
    
    with col2:
        st.markdown("**数学表达式**")
        st.latex(r'''
        z = W \cdot X + b
        ''')
        st.latex(r'''
        P(y=1|X) = \frac{1}{1 + e^{-z}}
        ''')
        st.latex(r'''
        \hat{y} = 
        \begin{cases} 
        1, & \text{if } P(y=1|X) > 0.5 \\
        0, & \text{otherwise}
        \end{cases}
        ''')
    
    # 交互式部分：权重影响
    st.markdown("### 交互体验：权重影响")
    
    st.markdown("""
    调整下面的滑块来观察特征权重如何影响逻辑回归的决策边界。
    简化起见，我们考虑一个二维特征空间和一个二分类问题。
    """)
    
    w1 = st.slider("特征1权重 (w₁)", -2.0, 2.0, 1.0, 0.1)
    w2 = st.slider("特征2权重 (w₂)", -2.0, 2.0, -1.0, 0.1)
    b = st.slider("偏置项 (b)", -3.0, 3.0, 0.0, 0.1)
    
    # 生成二维空间的数据点和决策边界
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 生成网格点
    x1_range = np.linspace(-5, 5, 100)
    x2_range = np.linspace(-5, 5, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    # 计算每个点的z值
    Z = w1 * X1 + w2 * X2 + b
    
    # 计算概率
    P = 1 / (1 + np.exp(-Z))
    
    # 绘制概率热图
    c = ax.contourf(X1, X2, P, cmap='coolwarm', alpha=0.6, levels=20)
    
    # 绘制决策边界
    ax.contour(X1, X2, P, levels=[0.5], colors='k', linewidths=2)
    
    # 标记轴
    ax.set_xlabel("特征1 (x₁)")
    ax.set_ylabel("特征2 (x₂)")
    ax.set_title("逻辑回归决策边界可视化")
    
    # 添加颜色条
    plt.colorbar(c, ax=ax, label="P(y=1|X)")
    
    st.pyplot(fig)
    
    st.markdown("""
    **注意事项**:
    - 黑色线表示决策边界，即P(y=1|X) = 0.5的位置
    - 红色区域表示模型更倾向于预测为正类(1)
    - 蓝色区域表示模型更倾向于预测为负类(0)
    - 调整权重和偏置可以改变决策边界的位置和方向
    """)

def show_svm_concept():
    """展示SVM的核心概念"""
    st.subheader("支持向量机(SVM)核心概念")
    
    st.markdown("""
    支持向量机(SVM)是一种强大的分类算法，它的核心思想是在特征空间中找到一个能够以最大间隔分隔不同类别的超平面。
    SVM特别关注那些最靠近决策边界的样本点，这些点被称为"支持向量"。
    """)
    
    # 显示SVM概念图 - 使用matplotlib代替SVG
    svm_fig = create_svm_concept_chart()
    st.pyplot(svm_fig)
    
    st.markdown("""
    **支持向量机的关键概念**:
    
    1. **最大间隔**：SVM寻找能够以最大间隔分隔两个类别的超平面，这有助于提高模型的泛化能力
    
    2. **支持向量**：支持向量是那些最靠近决策边界的数据点，它们"支持"或定义了最优决策边界的位置
    
    3. **核技巧**：通过使用核函数，SVM可以在不显式计算高维特征映射的情况下，在高维空间中建立非线性决策边界
    """)
    
    # 交互式部分：SVM参数影响
    st.markdown("### 交互体验：SVM参数影响")
    
    st.markdown("""
    调整下面的参数来观察它们如何影响SVM的决策边界和间隔。
    """)
    
    param_C = st.slider("正则化参数 (C)", 0.1, 10.0, 1.0, 0.1)
    kernel_type = st.selectbox("核函数类型", ["linear", "rbf", "poly"])
    
    if kernel_type == "rbf":
        gamma = st.slider("RBF核参数 (gamma)", 0.1, 5.0, 1.0, 0.1)
    elif kernel_type == "poly":
        degree = st.slider("多项式核次数", 1, 5, 3)
    
    # 生成示例数据
    np.random.seed(42)
    
    if st.checkbox("使用非线性可分数据", value=kernel_type != "linear"):
        # 生成同心圆数据
        n_samples = 100
        X1 = np.random.randn(n_samples, 2)
        X1 = X1 / np.linalg.norm(X1, axis=1).reshape(-1, 1) * np.random.uniform(0, 1, n_samples).reshape(-1, 1) * 2
        
        X2 = np.random.randn(n_samples, 2)
        X2 = X2 / np.linalg.norm(X2, axis=1).reshape(-1, 1) * np.random.uniform(1.5, 2.5, n_samples).reshape(-1, 1)
        
        X = np.vstack([X1, X2])
        y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    else:
        # 生成线性可分数据
        n_samples = 100
        X1 = np.random.randn(n_samples, 2) - np.array([2, 2])
        X2 = np.random.randn(n_samples, 2) + np.array([2, 2])
        X = np.vstack([X1, X2])
        y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    
    # 绘制数据点和SVM决策边界
    from sklearn.svm import SVC
    
    # 创建和训练SVM模型
    if kernel_type == "rbf":
        model = SVC(C=param_C, kernel=kernel_type, gamma=gamma)
    elif kernel_type == "poly":
        model = SVC(C=param_C, kernel=kernel_type, degree=degree)
    else:
        model = SVC(C=param_C, kernel=kernel_type)
    
    model.fit(X, y)
    
    # 绘制结果
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 创建网格以绘制决策边界
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # 预测网格中每个点的类别
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界和区域
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    ax.contour(xx, yy, Z, colors='k', linestyles=['-'], linewidths=[2])
    
    # 绘制支持向量
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
              linewidth=1, facecolors='none', edgecolors='k')
    
    # 绘制数据点
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel("特征1")
    ax.set_ylabel("特征2")
    ax.set_title(f"SVM决策边界 (C={param_C}, kernel={kernel_type})")
    
    st.pyplot(fig)
    
    st.markdown("""
    **说明**:
    
    - **黑色轮廓线**: 决策边界
    - **黑色空心圆圈**: 支持向量
    - **红色/蓝色点**: 不同类别的数据点
    - **C参数**: 控制错误分类的惩罚程度，较大的C会使模型尝试减少错误分类，可能导致过拟合
    - **核函数**: 
        - linear: 线性核，适用于线性可分数据
        - rbf: 径向基函数核，适用于非线性数据，gamma参数控制影响半径
        - poly: 多项式核，degree参数控制多项式次数
    """)

def show_model_comparison():
    """比较逻辑回归与SVM"""
    st.subheader("逻辑回归 vs 支持向量机(SVM)")
    
    # 显示模型比较图 - 使用matplotlib代替SVG
    comparison_fig = create_model_comparison_chart()
    st.pyplot(comparison_fig)
    
    st.markdown("""
    逻辑回归和SVM都是强大的分类算法，但它们有不同的特点和适用场景。上图展示了它们的详细比较。
    """)
    
    # 创建比较表格
    comparison_data = {
        "特性": [
            "基本原理", 
            "输出类型", 
            "决策边界", 
            "优化目标",
            "处理非线性",
            "计算复杂度",
            "处理大数据集",
            "处理高维数据",
            "处理小样本数据",
            "过拟合处理",
            "可解释性",
            "适用场景"
        ],
        "逻辑回归": [
            "通过Sigmoid函数将线性组合映射为概率",
            "概率值，范围在[0,1]之间",
            "线性决策边界(可通过特征工程扩展)",
            "最大化似然概率/最小化交叉熵损失",
            "需要显式特征转换(如多项式特征)",
            "相对较低，尤其对大数据集",
            "表现良好，可使用随机梯度下降",
            "容易过拟合，需要正则化",
            "样本过少时性能不佳",
            "L1/L2正则化",
            "高，权重直接反映特征重要性",
            "需要概率输出，需要解释性，大数据集"
        ],
        "支持向量机(SVM)": [
            "寻找最大间隔超平面分隔不同类别",
            "类别标签或到决策边界的距离",
            "取决于核函数，可以是线性或非线性",
            "最大化决策边界间隔/最小化结构风险",
            "通过核技巧隐式映射到高维空间",
            "较高，尤其对大数据集和非线性核",
            "线性SVM尚可，非线性SVM较慢",
            "处理高维数据的能力较强",
            "在小样本、高维数据上往往表现优异",
            "软间隔SVM(参数C)调整",
            "线性SVM可解释性好，核SVM较差",
            "小样本高维数据，需要非线性决策边界"
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df)
    
    # 交互式部分：不同数据集上的性能比较
    st.markdown("### 交互体验：在不同数据集上的表现比较")
    
    dataset_type = st.selectbox(
        "选择数据集类型",
        ["线性可分数据", "同心圆数据(非线性)", "噪声数据", "高度不平衡数据"]
    )
    
    # 生成不同类型的数据集
    np.random.seed(42)
    
    if dataset_type == "线性可分数据":
        # 线性可分数据
        n_samples = 100
        X1 = np.random.randn(n_samples, 2) - np.array([2, 2])
        X2 = np.random.randn(n_samples, 2) + np.array([2, 2])
        X = np.vstack([X1, X2])
        y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
        dataset_desc = "这是一个线性可分的数据集，两个类别可以通过一条直线清晰分开。"
        
    elif dataset_type == "同心圆数据(非线性)":
        # 同心圆数据
        n_samples = 100
        X1 = np.random.randn(n_samples, 2)
        X1 = X1 / np.linalg.norm(X1, axis=1).reshape(-1, 1) * np.random.uniform(0, 1, n_samples).reshape(-1, 1) * 2
        
        X2 = np.random.randn(n_samples, 2)
        X2 = X2 / np.linalg.norm(X2, axis=1).reshape(-1, 1) * np.random.uniform(1.5, 2.5, n_samples).reshape(-1, 1)
        
        X = np.vstack([X1, X2])
        y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
        dataset_desc = "这是一个非线性数据集，两个类别形成同心圆结构，无法通过直线分开。"
        
    elif dataset_type == "噪声数据":
        # 带噪声的数据
        n_samples = 100
        X1 = np.random.randn(n_samples, 2) - np.array([2, 2])
        X2 = np.random.randn(n_samples, 2) + np.array([2, 2])
        
        # 添加一些噪声点
        noise_points = 20
        noise_X = np.random.uniform(-4, 4, (noise_points, 2))
        noise_y = np.random.choice([0, 1], noise_points)
        
        X = np.vstack([X1, X2, noise_X])
        y = np.hstack([np.zeros(n_samples), np.ones(n_samples), noise_y])
        dataset_desc = "这是一个带有噪声的数据集，一些点的标签是随机的，使得完美分类变得困难。"
        
    else:  # 高度不平衡数据
        # 不平衡数据
        n_samples_majority = 150
        n_samples_minority = 20
        
        X1 = np.random.randn(n_samples_majority, 2) - np.array([2, 2])
        X2 = np.random.randn(n_samples_minority, 2) + np.array([2, 2])
        
        X = np.vstack([X1, X2])
        y = np.hstack([np.zeros(n_samples_majority), np.ones(n_samples_minority)])
        dataset_desc = "这是一个高度不平衡的数据集，一个类别的样本数量远多于另一个类别。"
    
    # 训练模型
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 逻辑回归
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    proba_lr = lr.predict_proba(X_test)[:, 1]
    
    # SVM
    if dataset_type == "同心圆数据(非线性)":
        # 非线性数据用RBF核
        svm = SVC(kernel='rbf', probability=True, random_state=42)
    else:
        # 其他数据用线性核
        svm = SVC(kernel='linear', probability=True, random_state=42)
    
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    proba_svm = svm.predict_proba(X_test)[:, 1]
    
    # 性能指标
    metrics = {
        "准确率(Accuracy)": [accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_svm)],
        "精确率(Precision)": [precision_score(y_test, y_pred_lr, zero_division=0), precision_score(y_test, y_pred_svm, zero_division=0)],
        "召回率(Recall)": [recall_score(y_test, y_pred_lr, zero_division=0), recall_score(y_test, y_pred_svm, zero_division=0)],
        "F1分数": [f1_score(y_test, y_pred_lr, zero_division=0), f1_score(y_test, y_pred_svm, zero_division=0)]
    }
    
    metrics_df = pd.DataFrame(metrics, index=["逻辑回归", "SVM"])
    
    # 可视化结果
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 创建网格
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # 逻辑回归决策边界
    Z_lr = lr.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_lr = Z_lr.reshape(xx.shape)
    axes[0].contourf(xx, yy, Z_lr, cmap=plt.cm.coolwarm, alpha=0.3)
    axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    axes[0].set_title("逻辑回归")
    
    # SVM决策边界
    Z_svm = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_svm = Z_svm.reshape(xx.shape)
    axes[1].contourf(xx, yy, Z_svm, cmap=plt.cm.coolwarm, alpha=0.3)
    axes[1].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    
    # 对于SVM，还显示支持向量
    if hasattr(svm, 'support_vectors_'):
        axes[1].scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], s=100,
                  linewidth=1, facecolors='none', edgecolors='k')
    
    axes[1].set_title("支持向量机(SVM)")
    
    # 设置坐标轴
    for ax in axes:
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel("特征1")
        ax.set_ylabel("特征2")
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 显示数据集描述
    st.markdown(f"**数据集描述**: {dataset_desc}")
    
    # 显示性能指标
    st.markdown("### 性能指标比较")
    st.dataframe(metrics_df)
    
    # 模型特点分析
    st.markdown("### 模型表现分析")
    
    if dataset_type == "线性可分数据":
        st.markdown("""
        在这个线性可分的数据集上:
        
        - **逻辑回归**表现良好，因为它是为线性决策边界设计的。决策边界是一条直线，能够有效分离两个类别。
        - **SVM**同样表现出色，线性核SVM能够找到最大间隔的分隔超平面。
        - 总体来说，当数据是线性可分的，两种算法都能取得不错的效果，但SVM可能略胜一筹，因为它专注于边界样本。
        """)
    
    elif dataset_type == "同心圆数据(非线性)":
        st.markdown("""
        在这个非线性数据集上:
        
        - **逻辑回归**表现较差，因为它默认情况下只能产生线性决策边界，无法适应同心圆的数据结构。
        - **SVM**使用RBF核后表现优异，能够创建非线性决策边界来分隔同心圆数据。
        - 这个例子突出了SVM通过核技巧处理非线性问题的强大能力，而逻辑回归需要显式的特征工程才能达到类似效果。
        """)
    
    elif dataset_type == "噪声数据":
        st.markdown("""
        在这个带噪声的数据集上:
        
        - **逻辑回归**对异常值相对敏感，可能会尝试调整决策边界以适应噪声点。
        - **SVM**通过控制软间隔参数C，可以在一定程度上忽略噪声点，产生更稳健的决策边界。
        - 在有噪声的实际应用中，两种算法都需要适当的正则化或其他技术来提高鲁棒性。
        """)
    
    else:  # 高度不平衡数据
        st.markdown("""
        在这个高度不平衡的数据集上:
        
        - **逻辑回归**可能偏向多数类，因为它的目标是最大化整体正确率。
        - **SVM**通常在不平衡数据上表现更好，特别是当你关注少数类时，因为它基于间隔而非整体错误率。
        - 对于不平衡数据，两种算法都可能需要额外技术如类别权重调整、过采样或欠采样等。
        """) 