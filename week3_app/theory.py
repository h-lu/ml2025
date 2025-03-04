import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import os

def show_theory():
    """显示分类算法的理论介绍"""
    
    st.header("理论介绍")
    
    tab1, tab2, tab3 = st.tabs(["逻辑回归", "支持向量机(SVM)", "机器学习基础"])
    
    with tab1:
        st.subheader("逻辑回归")
        
        st.markdown("""
        ### 概述
        
        逻辑回归是一种用于二分类问题的经典算法，它是线性回归的一种扩展，通过对线性模型的输出应用Sigmoid函数将输出映射到概率区间[0,1]。
        
        ### 数学原理
        
        线性回归模型表达式：$z = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n = w^Tx + b$
        
        逻辑回归通过Sigmoid函数将线性输出转化为概率：
        
        $$P(Y=1|X) = \\frac{1}{1 + e^{-z}} = \\frac{1}{1 + e^{-(w^Tx + b)}}$$
        
        这里的Sigmoid函数将任意实数值映射到(0,1)区间，表示样本属于正类的概率。
        """)
        
        # 显示Sigmoid函数图像
        st.markdown("#### Sigmoid函数图像")
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.linspace(-10, 10, 100)
        y = 1 / (1 + np.exp(-x))
        ax.plot(x, y)
        ax.grid(True)
        ax.set_xlabel("z")
        ax.set_ylabel("$\\sigma(z)$")
        ax.set_title("Sigmoid函数: $\\sigma(z) = \\frac{1}{1 + e^{-z}}$")
        st.pyplot(fig)
        
        st.markdown("""
        ### 损失函数
        
        逻辑回归使用对数似然损失函数(Log Likelihood Loss)，也称为交叉熵损失函数：
        
        $$J(w) = -\\frac{1}{m}\\sum_{i=1}^{m} [y^{(i)}\\log(h_w(x^{(i)})) + (1-y^{(i)})\\log(1-h_w(x^{(i)}))]$$
        
        其中，$h_w(x^{(i)})$是模型对第$i$个样本预测为正类的概率。
        
        ### 优点与局限性
        
        **优点：**
        - 简单易实现，计算效率高
        - 可解释性强，权重直接反映特征重要性
        - 不易过拟合，泛化能力较好
        - 可以输出概率值而非仅分类标签
        
        **局限性：**
        - 只能处理线性可分的问题
        - 对特征共线性较为敏感
        - 对异常值敏感
        """)
    
    with tab2:
        st.subheader("支持向量机(SVM)")
        
        st.markdown("""
        ### 概述
        
        支持向量机是一种强大的分类算法，目标是找到一个超平面，使得它能够最大化不同类别样本之间的间隔。SVM特别适合处理高维数据，并且通过核技巧可以解决非线性分类问题。
        
        ### 数学原理
        
        线性SVM的核心是找到一个超平面：$w^T x + b = 0$，使得：
        
        $$\\min_{w,b} \\frac{1}{2}||w||^2$$
        $$s.t. y_i(w^T x_i + b) \\geq 1, \\forall i = 1, 2, ..., m$$
        
        这里的约束保证每个样本到超平面的距离至少为$\\frac{1}{||w||}$，而目标是最大化这个距离，即最小化$||w||^2$。
        """)
        
        st.markdown("#### 超平面与支持向量示意图")
        # 使用相对于当前脚本的路径
        image_path = os.path.join(os.path.dirname(__file__), "img", "week3", "svm_hyperplane.png")
        st.image(image_path, caption="支持向量机的超平面与支持向量", use_column_width=True)
        
        st.markdown("""
        ### 核技巧
        
        对于非线性问题，SVM使用核函数将数据映射到高维空间，使其在高维空间中线性可分：
        
        - **线性核**: $K(x_i, x_j) = x_i^T x_j$
        - **多项式核**: $K(x_i, x_j) = (\\gamma x_i^T x_j + r)^d$
        - **RBF核(高斯核)**: $K(x_i, x_j) = \\exp(-\\gamma ||x_i - x_j||^2)$
        
        ### 硬间隔与软间隔
        
        - **硬间隔**: 要求所有样本都必须正确分类，不允许任何错误
        - **软间隔**: 允许一些样本分类错误，但会给予惩罚，引入松弛变量$\\xi_i$和惩罚参数$C$
        
        ### 优点与局限性
        
        **优点：**
        - 在高维空间中效果好
        - 能够处理非线性决策边界
        - 较强的泛化能力
        - 不易受异常值影响（软间隔SVM）
        
        **局限性：**
        - 计算复杂度高，不适合大规模数据
        - 对参数敏感，需要细致调优
        - 不直接提供概率估计
        """)
    
    with tab3:
        st.subheader("机器学习基础概念")
        
        st.markdown("""
        本节介绍机器学习中的一些关键基础概念，这些概念对理解分类算法至关重要。
        
        ### 过拟合与欠拟合
        
        **过拟合(Overfitting)**是指模型在训练数据上表现良好，但在新数据上表现不佳的现象。过拟合的模型"记住"了训练数据中的噪声和随机波动，而不是学习到数据的真实模式。
        
        **欠拟合(Underfitting)**是指模型既不能很好地拟合训练数据，也不能很好地泛化到新数据的情况。欠拟合通常是由于模型过于简单，无法捕捉数据中的复杂模式所致。
        
        下图直观地展示了欠拟合、适当拟合和过拟合：
        """)
        
        # 绘制过拟合vs欠拟合示意图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 生成示例数据
        np.random.seed(42)
        x = np.linspace(0, 1, 30)
        y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, 30)
        x_plot = np.linspace(0, 1, 100)
        y_real = np.sin(2 * np.pi * x_plot)
        
        # 欠拟合（线性模型）
        lr = LinearRegression()
        lr.fit(x.reshape(-1, 1), y)
        y_pred_lr = lr.predict(x_plot.reshape(-1, 1))
        
        axes[0].scatter(x, y, color='blue', label='数据点')
        axes[0].plot(x_plot, y_real, 'g--', label='真实函数')
        axes[0].plot(x_plot, y_pred_lr, 'r-', label='线性模型')
        axes[0].set_title('欠拟合')
        axes[0].legend()
        axes[0].set_ylim(-1.5, 1.5)
        
        # 适当拟合（三次多项式）
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=3)),
            ('linear', LinearRegression())
        ])
        model.fit(x.reshape(-1, 1), y)
        y_pred_model = model.predict(x_plot.reshape(-1, 1))
        
        axes[1].scatter(x, y, color='blue', label='数据点')
        axes[1].plot(x_plot, y_real, 'g--', label='真实函数')
        axes[1].plot(x_plot, y_pred_model, 'r-', label='三次多项式')
        axes[1].set_title('适当拟合')
        axes[1].legend()
        axes[1].set_ylim(-1.5, 1.5)
        
        # 过拟合（高次多项式）
        model_overfit = Pipeline([
            ('poly', PolynomialFeatures(degree=15)),
            ('linear', LinearRegression())
        ])
        model_overfit.fit(x.reshape(-1, 1), y)
        y_pred_overfit = model_overfit.predict(x_plot.reshape(-1, 1))
        
        axes[2].scatter(x, y, color='blue', label='数据点')
        axes[2].plot(x_plot, y_real, 'g--', label='真实函数')
        axes[2].plot(x_plot, y_pred_overfit, 'r-', label='15次多项式')
        axes[2].set_title('过拟合')
        axes[2].legend()
        axes[2].set_ylim(-1.5, 1.5)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        ### 偏差-方差权衡
        
        模型的预测误差可以分解为三个关键组成部分：
        
        1. **偏差(Bias)** - 模型预测值与真实值的平均差异。高偏差模型通常过于简单，无法捕捉数据中的复杂模式，导致欠拟合。
        
        2. **方差(Variance)** - 对不同训练集的敏感度。高方差模型对训练数据中的微小变化非常敏感，容易过拟合。
        
        3. **不可约误差** - 数据本身的噪声，无法通过任何模型消除。
        
        总误差 = 偏差² + 方差 + 不可约误差
        
        模型复杂度增加时，偏差通常会减少，而方差会增加。模型设计的挑战是找到平衡点，即**最小化总误差**。
        """)
        
        # 绘制偏差-方差权衡图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        model_complexity = np.linspace(1, 10, 100)
        bias = 10 / model_complexity
        variance = 0.1 * model_complexity
        total_error = bias + variance + 1
        
        ax.plot(model_complexity, bias, 'b-', label='偏差')
        ax.plot(model_complexity, variance, 'r-', label='方差')
        ax.plot(model_complexity, total_error, 'g-', label='总误差')
        ax.axvline(x=np.sqrt(10/0.1), color='k', linestyle='--', label='最优复杂度')
        
        ax.set_xlabel('模型复杂度')
        ax.set_ylabel('误差')
        ax.set_title('偏差-方差权衡')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        st.markdown("""
        ### 正则化：控制模型复杂度
        
        **正则化**是一种防止过拟合的技术，通过向损失函数添加惩罚项来限制模型参数的大小，从而降低模型复杂度。
        
        #### 为什么需要正则化？
        
        - 防止模型过度拟合训练数据
        - 提高模型泛化能力
        - 处理高维数据中的特征共线性
        - 在有大量特征但少量样本的情况下尤其重要
        
        #### 常用的正则化方法：
        
        1. **L1正则化(Lasso)**：向损失函数添加参数绝对值之和的惩罚
           - 损失函数：$L(w) + \lambda \sum_{i=1}^{n} |w_i|$
           - 特点：倾向于产生稀疏解（许多参数为零），可用于特征选择
           
        2. **L2正则化(Ridge)**：向损失函数添加参数平方和的惩罚
           - 损失函数：$L(w) + \lambda \sum_{i=1}^{n} w_i^2$
           - 特点：惩罚较大的参数，使所有参数值变小但不为零
           
        3. **弹性网络(Elastic Net)**：结合L1和L2正则化
           - 损失函数：$L(w) + \lambda_1 \sum_{i=1}^{n} |w_i| + \lambda_2 \sum_{i=1}^{n} w_i^2$
           - 特点：结合了Lasso和Ridge的优点
        
        #### 正则化强度：
        
        正则化强度参数$\lambda$（在scikit-learn中常表示为C的倒数）控制了惩罚的程度：
        - 较大的$\lambda$：更强的正则化，模型更简单，可能欠拟合
        - 较小的$\lambda$：更弱的正则化，模型更复杂，可能过拟合
        
        选择适当的正则化强度通常需要通过交叉验证来确定。
        """)
        
        st.markdown("""
        ### 超参数选择
        
        **超参数**是在模型训练开始前设置的参数，不同于模型在训练过程中学习的参数。对于分类算法，常见的超参数包括：
        
        - 逻辑回归：正则化强度C，正则化类型
        - SVM：正则化强度C，核函数类型，核函数参数(如gamma)
        
        #### 如何选择超参数？
        
        1. **网格搜索(Grid Search)**：
           - 定义超参数的可能值网格
           - 对每种组合训练和评估模型
           - 选择性能最佳的组合
           
        2. **随机搜索(Random Search)**：
           - 从超参数的分布中随机采样
           - 适用于超参数空间较大的情况
           
        3. **贝叶斯优化**：
           - 基于先前的评估结果智能地选择下一组参数
           - 通常比网格和随机搜索更高效
        
        #### 数据集划分
        
        合理划分数据集对于超参数选择和模型评估至关重要：
        
        1. **训练集(Training Set)**：
           - 用于训练模型，模型直接学习这些数据
           - 通常占总数据的60-80%
           
        2. **验证集(Validation Set)**：
           - 用于超参数调优和模型选择
           - 不直接参与模型训练
           - 通常占总数据的10-20%
           
        3. **测试集(Test Set)**：
           - 用于评估最终模型性能
           - 只使用一次，代表"真实世界"的数据
           - 通常占总数据的10-20%
        
        4. **交叉验证(Cross-Validation)**：
           - 当数据量有限时特别有用
           - k折交叉验证：将数据分成k份，轮流使用其中一份作为验证集，其余作为训练集
           - 提供更稳健的性能估计
        """)
        
        # 绘制数据集划分示意图
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # 创建一个表示整个数据集的矩形
        ax.add_patch(plt.Rectangle((0, 0), 100, 1, fc='lightgray', ec='black'))
        
        # 划分区域
        ax.add_patch(plt.Rectangle((0, 0), 70, 1, fc='lightblue', ec='black'))
        ax.add_patch(plt.Rectangle((70, 0), 15, 1, fc='lightgreen', ec='black'))
        ax.add_patch(plt.Rectangle((85, 0), 15, 1, fc='salmon', ec='black'))
        
        # 添加标签
        ax.text(35, 0.5, '训练集 (70%)', ha='center', va='center', fontsize=12)
        ax.text(77.5, 0.5, '验证集\n(15%)', ha='center', va='center', fontsize=12)
        ax.text(92.5, 0.5, '测试集\n(15%)', ha='center', va='center', fontsize=12)
        
        # 设置轴
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('数据集划分示意图', fontsize=14)
        
        st.pyplot(fig)
        
        st.markdown("""
        ### 评估分类模型性能
        
        选择合适的评估指标对于理解模型性能至关重要：
        
        1. **准确率(Accuracy)**：正确预测的比例
           - 适用于平衡数据集
           - 公式：(TP + TN) / (TP + TN + FP + FN)
           
        2. **精确率(Precision)**：预测为正的样本中实际为正的比例
           - 衡量正类预测的准确性
           - 公式：TP / (TP + FP)
           
        3. **召回率(Recall)**：实际为正的样本中被正确预测的比例
           - 衡量模型发现正类的能力
           - 公式：TP / (TP + FN)
           
        4. **F1分数**：精确率和召回率的调和平均
           - 公式：2 * (Precision * Recall) / (Precision + Recall)
           
        5. **ROC曲线和AUC**：
           - 反映模型在不同决策阈值下的表现
           - AUC(曲线下面积)越接近1越好
        
        选择合适的评估指标取决于具体应用场景和错误的相对成本。例如，在医疗诊断中，高召回率(避免漏诊)可能比高精确率更重要。
        """)
    
    st.markdown("### 两种算法的比较")
    
    comparison_data = {
        "特性": ["模型类型", "决策边界", "计算复杂度", "可解释性", "处理非线性能力", "大规模数据表现", "概率输出"],
        "逻辑回归": ["判别模型", "线性", "低", "高", "弱(需特征工程)", "好", "直接输出概率"],
        "SVM": ["判别模型", "线性或非线性", "中至高", "中(线性核高)", "强(使用核函数)", "较差", "需额外计算"]
    }
    
    st.table(comparison_data)
    
    st.markdown("""
    ### 何时选择哪种算法？
    
    - **选择逻辑回归**：
      - 需要模型可解释性
      - 数据规模大
      - 需要直接获得概率输出
      - 特征间关系较为简单，近似线性可分
    
    - **选择SVM**：
      - 处理高维数据
      - 数据集中等规模
      - 需要处理非线性决策边界
      - 准确率是首要考虑因素
    """) 