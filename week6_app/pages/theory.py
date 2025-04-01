import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# 导入matplotlib中文字体支持
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.plot_utils import configure_matplotlib_fonts

# 配置matplotlib支持中文
configure_matplotlib_fonts()

def show():
    st.title("集成学习基础理论")
    
    # 侧边栏选择
    topic = st.sidebar.radio(
        "选择主题",
        ["集成学习简介", "GBDT基础", "XGBoost详解", "XGBoost模型比较", "Stacking集成", "评估指标"]
    )
    
    if topic == "集成学习简介":
        show_ensemble_learning()
    elif topic == "GBDT基础":
        show_gbdt_basics()
    elif topic == "XGBoost详解":
        show_xgboost_details()
    elif topic == "XGBoost模型比较":
        show_xgboost_comparison()
    elif topic == "Stacking集成":
        show_stacking()
    elif topic == "评估指标":
        show_evaluation_metrics()

def show_ensemble_learning():
    st.header("集成学习概述")
    
    # 理论解释
    st.markdown("""
    ### 什么是集成学习？
    
    集成学习是通过组合多个基学习器来提高学习效果的一种机器学习方法。相比单一模型，集成模型往往能够获得更好的预测性能。
    """)
    
    # 主要策略
    st.subheader("主要策略")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Bagging")
        st.info("""
        **原理**：并行训练多个基学习器
        
        **代表算法**：随机森林(Random Forest)
        
        **特点**：
        - 每个基学习器使用原始数据的随机子集
        - 基学习器之间相互独立
        - 通过投票或平均减少方差
        """)
    
    with col2:
        st.markdown("#### Boosting")
        st.info("""
        **原理**：串行训练基学习器
        
        **代表算法**：AdaBoost, GBDT, XGBoost
        
        **特点**：
        - 每个基学习器关注前一个学习器的错误
        - 基学习器之间依赖
        - 通过加权组合减少偏差
        """)
    
    with col3:
        st.markdown("#### Stacking")
        st.info("""
        **原理**：使用新模型组合多个模型
        
        **特点**：
        - 训练多个不同类型的基学习器
        - 使用元学习器整合基学习器的输出
        - 能组合不同类型的模型优势
        """)
    
    # 优势
    st.subheader("集成学习的优势")
    st.markdown("""
    * **降低方差**：减少过拟合风险
    * **提高模型稳定性**：减少随机性影响
    * **提高预测准确性**：组合多个模型的优势
    * **处理复杂非线性关系**：强大的表达能力
    """)
    
    # 可视化示例
    st.subheader("集成学习效果示意图")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # 模拟数据
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y_true = np.sin(x) + 0.3*x
    y_noise = y_true + np.random.normal(0, 0.5, size=len(x))
    
    # 弱学习器预测 - 使每个学习器都有明显的缺陷但在不同区域表现较好
    weak_1 = y_true + 0.7 * np.sin(x/2) # 在某些区域偏高
    weak_2 = y_true - 0.7 * np.cos(x/2) # 在某些区域偏低
    
    # 集成结果 - 两个弱学习器取平均，消除各自的缺陷
    ensemble = (weak_1 + weak_2) / 2
    
    # 计算均方误差，用于量化比较
    mse_weak1 = np.mean((weak_1 - y_true) ** 2)
    mse_weak2 = np.mean((weak_2 - y_true) ** 2)
    mse_ensemble = np.mean((ensemble - y_true) ** 2)
    
    # 绘图
    ax[0].scatter(x, y_noise, alpha=0.5, label='数据点')
    ax[0].plot(x, y_true, 'r-', label='真实关系')
    ax[0].plot(x, weak_1, 'g--', label='弱学习器1')
    ax[0].set_title(f'弱学习器1 (MSE: {mse_weak1:.2f})')
    ax[0].legend()
    
    ax[1].scatter(x, y_noise, alpha=0.5, label='数据点')
    ax[1].plot(x, y_true, 'r-', label='真实关系')
    ax[1].plot(x, weak_2, 'b--', label='弱学习器2')
    ax[1].set_title(f'弱学习器2 (MSE: {mse_weak2:.2f})')
    ax[1].legend()
    
    ax[2].scatter(x, y_noise, alpha=0.5, label='数据点')
    ax[2].plot(x, y_true, 'r-', label='真实关系')
    ax[2].plot(x, ensemble, 'y-', label='集成结果')
    ax[2].set_title(f'集成模型 (MSE: {mse_ensemble:.2f})')
    ax[2].legend()
    
    st.pyplot(fig)
    
    # 添加明确的性能比较
    st.markdown("""
    ### 集成学习的性能提升
    
    上图展示了集成学习如何结合多个弱学习器的优势：
    
    | 模型 | 均方误差 (MSE) | 提升百分比 |
    |------|---------------|-----------|
    | 弱学习器1 | {:.2f} | - |
    | 弱学习器2 | {:.2f} | - |
    | 集成模型 | {:.2f} | {:.1f}% |
    
    集成模型相比平均单一模型误差降低了{:.1f}%。这展示了集成学习的核心优势：**通过组合多个模型可以抵消个体模型的缺陷，产生更稳定、更准确的预测**。
    """.format(
        mse_weak1, 
        mse_weak2, 
        mse_ensemble, 
        ((mse_weak1 + mse_weak2) / 2 - mse_ensemble) / ((mse_weak1 + mse_weak2) / 2) * 100,
        ((mse_weak1 + mse_weak2) / 2 - mse_ensemble) / ((mse_weak1 + mse_weak2) / 2) * 100
    ))
    
    # 交互问题
    st.subheader("思考问题")
    q1 = st.radio(
        "集成学习中，为什么多个'弱学习器'的组合能够产生一个'强学习器'？",
        ["偏差-方差权衡", "增加计算复杂度", "利用了数据的多个副本", "我不确定"]
    )
    
    if q1 == "偏差-方差权衡":
        st.success("正确！集成学习通过组合多个模型来降低整体方差，同时通过迭代学习降低偏差。")
    elif q1 != "我不确定":
        st.error("再考虑一下？集成学习的核心优势在于如何平衡偏差和方差。")

def show_gbdt_basics():
    st.header("梯度提升决策树(Gradient Boosting Decision Tree)基础")
    
    st.markdown("""
    ### 基本思想
    
    梯度提升决策树(GBDT)是Boosting家族中的一员，其核心思想是：**通过不断拟合前一个模型的残差来提高整体模型性能**。
    
    ### GBDT简明解释
    
    想象你在预测房价，但结果不够准确:
    
    1. **第一步**: 先用一个简单模型做初步预测（比如平均房价）
    2. **第二步**: 计算预测误差（残差）
    3. **第三步**: 训练一个新模型专门预测这些误差
    4. **第四步**: 将误差预测和原始预测相加，得到更准确的结果
    5. **重复步骤2-4**: 一直迭代，每次都训练新模型来修正之前预测的错误
    
    通过这种方式，GBDT通过"**一点一点**"地修正误差，逐步提高整体预测精度。
    """)
    
    # 交互式展示GBDT过程
    st.subheader("GBDT的工作流程")
    
    # 创建一个更多的交互元素
    n_trees = st.slider("决策树数量", 1, 10, 3)
    learning_rate = st.slider("学习率", 0.1, 1.0, 0.3, 0.1)
    
    if st.button("展示GBDT迭代过程"):
        display_gbdt_process(n_trees, learning_rate)
    
    # GBDT算法步骤
    st.subheader("GBDT算法步骤")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        #### 算法流程
        
        1. **初始化模型 $F_0(x)$** = 一个常数值(通常是数据的平均值)
        
        2. **对每棵树 $m = 1 to M$**:
        
           a. 计算**负梯度/残差**: 
              $$r_{im} = y_i - F_{m-1}(x_i)$$
           
           b. 训练一棵决策树 $h_m(x)$ 来预测这些残差
           
           c. 使用学习率 $\eta$ 更新模型: 
              $$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$
        
        3. **最终模型**: 
           $$F_M(x) = F_0(x) + \eta\sum_{m=1}^{M} h_m(x)$$
        """)
    
    with col2:
        st.markdown("""
        #### 关键点解释
        
        - **残差**: 实际值与当前预测值的差距，表示"还有多少没学会"
        
        - **决策树**: 每棵树专注于修正前一个模型的错误
        
        - **学习率**: 控制每棵树的贡献力度，防止过拟合
          - 较小的学习率 → 更平滑的学习过程，但需要更多树
          - 较大的学习率 → 学习更快，但可能过拟合
        
        - **累加效果**: 最终模型是所有树的预测加权累加
        """)
    
    # 可视化GBDT与其他模型的比较
    st.subheader("GBDT与其他模型的比较")
    
    # 生成示例数据
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y_true = 2 + 0.5 * X.ravel() + np.sin(X.ravel()) + np.random.normal(0, 0.5, X.shape[0])
    
    # 拟合不同模型
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    
    X_test = np.linspace(0, 10, 300).reshape(-1, 1)
    
    # 线性回归
    lr = LinearRegression()
    lr.fit(X, y_true)
    y_lr = lr.predict(X_test)
    
    # 单一决策树
    tree = DecisionTreeRegressor(max_depth=3)
    tree.fit(X, y_true)
    y_tree = tree.predict(X_test)
    
    # GBDT
    gbdt = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    gbdt.fit(X, y_true)
    y_gbdt = gbdt.predict(X_test)
    
    # 计算MSE
    mse_lr = np.mean((lr.predict(X) - y_true) ** 2)
    mse_tree = np.mean((tree.predict(X) - y_true) ** 2)
    mse_gbdt = np.mean((gbdt.predict(X) - y_true) ** 2)
    
    # 绘图
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    ax[0].scatter(X, y_true, alpha=0.5, label='数据点')
    ax[0].plot(X_test, y_lr, 'r-', label='线性回归')
    ax[0].set_title(f'线性回归 (MSE: {mse_lr:.2f})')
    ax[0].legend()
    
    ax[1].scatter(X, y_true, alpha=0.5, label='数据点')
    ax[1].plot(X_test, y_tree, 'g-', label='单一决策树')
    ax[1].set_title(f'单一决策树 (MSE: {mse_tree:.2f})')
    ax[1].legend()
    
    ax[2].scatter(X, y_true, alpha=0.5, label='数据点')
    ax[2].plot(X_test, y_gbdt, 'b-', label='GBDT')
    ax[2].set_title(f'GBDT (MSE: {mse_gbdt:.2f})')
    ax[2].legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 性能对比表格
    st.markdown("""
    | 模型 | 均方误差 (MSE) | 提升百分比 |
    |------|---------------|-----------|
    | 线性回归 | {:.2f} | - |
    | 单一决策树 | {:.2f} | - |
    | GBDT | {:.2f} | {:.1f}% |
    
    GBDT相比线性回归和单一决策树有明显的性能提升，尤其在处理**非线性关系**时优势更明显。
    """.format(
        mse_lr, 
        mse_tree, 
        mse_gbdt, 
        (min(mse_lr, mse_tree) - mse_gbdt) / min(mse_lr, mse_tree) * 100
    ))

def display_gbdt_process(n_trees, learning_rate):
    """
    显示GBDT迭代过程的可视化
    """
    # 模拟数据
    np.random.seed(42)
    x = np.linspace(0, 10, 100).reshape(-1, 1)
    y_true = np.sin(x.ravel()) + 0.3*x.ravel()
    y_noise = y_true + np.random.normal(0, 0.5, size=len(x))
    
    # 绘制GBDT迭代过程
    # 初始预测为均值
    f0 = np.mean(y_noise) * np.ones_like(y_noise)
    residuals = y_noise - f0
    
    fig, axes = plt.subplots(2, n_trees, figsize=(15, 10))
    
    # 如果只有一棵树，调整axes的维度
    if n_trees == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    
    # 第一行展示每个树的拟合
    axes[0, 0].scatter(x, y_noise, alpha=0.5, label='数据点')
    axes[0, 0].axhline(np.mean(y_noise), color='g', linestyle='-', label='平均值预测')
    axes[0, 0].set_title('初始模型 (常数预测)')
    axes[0, 0].legend()
    
    # 第二行展示累积效果
    axes[1, 0].scatter(x, y_noise, alpha=0.5, label='数据点')
    axes[1, 0].axhline(np.mean(y_noise), color='g', linestyle='-', label='平均值预测')
    axes[1, 0].plot(x, y_true, 'b--', label='真实关系')
    axes[1, 0].set_title('初始预测')
    axes[1, 0].legend()
    
    # 简化的树拟合函数 - 使用更复杂的树模型
    def simple_tree_predict(x, residuals, depth=3):
        """简单的决策树实现，递归分裂数据"""
        if depth == 0 or len(x) <= 5:
            return np.mean(residuals) * np.ones_like(residuals)
        
        # 找最佳分割点
        best_split = np.median(x)
        left_mask = x.ravel() <= best_split
        right_mask = ~left_mask
        
        # 如果分割后任一侧样本太少，返回均值预测
        if np.sum(left_mask) <= 2 or np.sum(right_mask) <= 2:
            return np.mean(residuals) * np.ones_like(residuals)
        
        # 递归构建左右子树
        predictions = np.zeros_like(residuals)
        if np.sum(left_mask) > 0:
            left_pred = simple_tree_predict(x[left_mask], residuals[left_mask], depth-1)
            predictions[left_mask] = left_pred
        
        if np.sum(right_mask) > 0:
            right_pred = simple_tree_predict(x[right_mask], residuals[right_mask], depth-1)
            predictions[right_mask] = right_pred
        
        return predictions
    
    # 当前预测
    current_pred = f0.copy()
    
    # 按顺序生成每棵树并展示
    for i in range(1, n_trees):
        # 计算残差
        residuals = y_noise - current_pred
        
        # 拟合树
        tree_pred = simple_tree_predict(x, residuals)
        
        # 更新预测
        current_pred += learning_rate * tree_pred
        
        # 计算当前MSE
        current_mse = np.mean((current_pred - y_true)**2)
        
        # 绘制每个树的拟合
        axes[0, i].scatter(x, residuals, alpha=0.5, label='残差')
        axes[0, i].plot(x, tree_pred, 'g-', label=f'树{i}预测')
        axes[0, i].set_title(f'树 {i} 拟合残差')
        axes[0, i].legend()
        
        # 绘制累积效果
        axes[1, i].scatter(x, y_noise, alpha=0.5, label='数据点')
        axes[1, i].plot(x, current_pred, 'r-', label=f'累积预测')
        axes[1, i].plot(x, y_true, 'b--', label='真实关系')
        axes[1, i].set_title(f'{i}棵树后的预测 (MSE: {current_mse:.2f})')
        axes[1, i].legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 添加解释性文字
    st.markdown("""
    ### GBDT迭代过程解释
    
    1. **第一行图像**：显示每棵树拟合的残差。每棵新树都在学习前面模型尚未学到的模式。
    
    2. **第二行图像**：显示模型累积效果。随着树的增加，模型预测越来越接近真实关系。
    
    3. **学习率({:.1f})**：控制每棵树的贡献权重，较小的学习率可能需要更多的树。
    
    4. **性能提升**：注意MSE(均方误差)随着树数量增加而降低，显示了模型不断改进的过程。
    """.format(learning_rate))

def show_xgboost_details():
    st.header("XGBoost算法详解")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### XGBoost简介
        
        XGBoost (eXtreme Gradient Boosting) 是GBDT的高效实现和扩展，由陈天奇等人开发，于2014年发布。它在各类机器学习竞赛中表现出色，成为数据科学家最常用的算法之一。
        
        ### XGBoost vs 传统GBDT
        
        XGBoost对传统GBDT进行了多方面的改进:
        
        1. **正则化**：添加了L1和L2正则项，减少过拟合
        
        2. **并行计算**：利用多核CPU进行并行特征计算
        
        3. **缺失值处理**：内置机制自动处理缺失值
        
        4. **二阶导数**：使用目标函数的一阶和二阶导数进行更精确的优化
        
        5. **剪枝**：使用"最大深度"限制和后剪枝技术
        
        6. **交叉验证**：内置交叉验证功能
        """)
    
    with col2:
        st.markdown("""
        ### 核心参数
        
        XGBoost有很多参数，以下是最重要的几个：
        
        | 参数 | 描述 | 典型值 |
        |------|------|-------|
        | n_estimators | 树的数量 | 100-1000 |
        | learning_rate | 学习率 | 0.01-0.3 |
        | max_depth | 树的最大深度 | 3-10 |
        | min_child_weight | 最小子节点权重 | 1-10 |
        | subsample | 训练样本抽样比例 | 0.5-1.0 |
        | colsample_bytree | 特征抽样比例 | 0.5-1.0 |
        | reg_alpha | L1正则化项 | 0-1.0 |
        | reg_lambda | L2正则化项 | 1.0 |
        """)
    
    # XGBoost数学原理
    st.subheader("XGBoost的数学原理")
    st.markdown("""
    ### 基本目标函数
    
    XGBoost使用一阶和二阶导数来优化目标函数：
    
    $$\mathcal{L}(\phi) = \sum_{i=1}^n l(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)$$
    
    其中:
    - $l(y_i, \hat{y}_i)$ 是损失函数，衡量预测值与实际值的偏差
    - $\Omega(f_k)$ 是第k棵树的复杂度惩罚项，定义为：$\\Omega(f) = \\gamma T + \\frac{1}{2}\\lambda \\sum_{j=1}^T w_j^2$
    - $\hat{y}_i = \sum_{k=1}^K f_k(x_i)$ 是预测值，即所有K棵树的输出总和
    - $T$ 是叶子节点数量，$w_j$ 是第j个叶子节点的权重
    
    ### 泰勒展开优化
    
    XGBoost使用泰勒展开对目标函数进行二阶近似，让优化更高效：
    
    $$\\mathcal{L}^{(t)} \\approx \\sum_{i=1}^n [l(y_i, \\hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \\frac{1}{2} h_i f_t^2(x_i)] + \\Omega(f_t)$$
    
    其中:
    - $g_i = \partial_{\hat{y}_i^{(t-1)}} l(y_i, \hat{y}_i^{(t-1)})$ 是一阶导数（梯度）
    - $h_i = \partial_{\hat{y}_i^{(t-1)}}^2 l(y_i, \hat{y}_i^{(t-1)})$ 是二阶导数（海森矩阵）
    - $\hat{y}_i^{(t-1)}$ 是第t-1轮的预测结果
    
    ### 树结构优化
    
    对于单棵树，去掉常数项后，目标函数可简化为：
    
    $$\\tilde{\mathcal{L}}^{(t)} = \\sum_{i=1}^n [g_i f_t(x_i) + \\frac{1}{2} h_i f_t^2(x_i)] + \\Omega(f_t)$$
    
    对于树的结构，定义$I_j$为落在叶子节点j上的样本集合，则目标函数可重写为：
    
    $$\tilde{\mathcal{L}}^{(t)} = \sum_{j=1}^T [(\sum_{i \in I_j} g_i) w_j + \frac{1}{2} (\sum_{i \in I_j} h_i + \lambda) w_j^2] + \gamma T$$
    
    ### 最优叶子权重
    
    对权重$w_j$求导并令其为0，可得最优叶子节点权重：
    
    $$w_j^* = -\\frac{\\sum_{i \\in I_j} g_i}{\\sum_{i \\in I_j} h_i + \\lambda}$$
    
    ### 节点分裂增益
    
    节点分裂的增益计算公式为：
    
    $$Gain = \\frac{1}{2} \\left[ \\frac{(\\sum_{i \\in I_L} g_i)^2}{\\sum_{i \\in I_L} h_i + \\lambda} + \\frac{(\\sum_{i \\in I_R} g_i)^2}{\\sum_{i \\in I_R} h_i + \\lambda} - \\frac{(\\sum_{i \\in I} g_i)^2}{\\sum_{i \\in I} h_i + \\lambda} \\right] - \\gamma$$
    
    其中：
    - $I_L$ 和 $I_R$ 分别是分裂后左右节点的样本集合
    - $\gamma$ 是阈值参数，用于控制分裂操作
    
    ### 不同损失函数的梯度和海森矩阵
    
    1. **均方误差（回归）**:
       - 损失函数：$l(y_i, \hat{y}_i) = \\frac{1}{2}(y_i - \hat{y}_i)^2$
       - 一阶导数：$g_i = \hat{y}_i - y_i$
       - 二阶导数：$h_i = 1$
    
    2. **对数损失（二分类）**:
       - 损失函数：$l(y_i, \\hat{y}_i) = y_i \\ln(1+e^{-\\hat{y}_i}) + (1-y_i) \\ln(1+e^{\\hat{y}_i})$
       - 一阶导数：$g_i = \\frac{e^{\\hat{y}_i}}{1+e^{\\hat{y}_i}} - y_i$
       - 二阶导数：$h_i = \\frac{e^{\\hat{y}_i}}{(1+e^{\\hat{y}_i})^2}$
    
    XGBoost强大的性能很大程度上来自于这种高效的目标函数优化方法和灵活的正则化策略。
    """)
    
    # 添加XGBoost迭代过程可视化
    st.subheader("XGBoost迭代学习过程可视化")
    
    col1, col2 = st.columns(2)
    with col1:
        n_trees_xgb = st.slider("树的数量", 1, 200, 10)
        learning_rate_xgb = st.slider("学习率", 0.01, 1.0, 0.1, 0.01)
        max_depth_xgb = st.slider("树的最大深度", 1, 10, 3)
    
    with col2:
        subsample_xgb = st.slider("样本采样比例", 0.1, 1.0, 1.0, 0.1)
        colsample_bytree_xgb = st.slider("特征采样比例", 0.1, 1.0, 1.0, 0.1)
        reg_lambda_xgb = st.slider("L2正则化系数", 0.0, 10.0, 1.0, 0.1)
    
    if st.button("展示XGBoost迭代过程"):
        # 显示XGBoost的迭代学习过程
        display_xgboost_process(
            n_trees=n_trees_xgb, 
            learning_rate=learning_rate_xgb,
            max_depth=max_depth_xgb,
            subsample=subsample_xgb,
            colsample_bytree=colsample_bytree_xgb,
            reg_lambda=reg_lambda_xgb
        )

    # XGBoost应用案例
    st.subheader("XGBoost的应用场景")
    st.markdown("""
    XGBoost广泛应用于:
    
    - **结构化数据预测**：表格数据的分类和回归问题
    - **金融风险建模**：信用评分、欺诈检测
    - **推荐系统**：预测用户行为和偏好
    - **医疗诊断**：疾病风险预测
    - **广告点击率预测**：CTR预测
    
    XGBoost特别适合处理**中等规模的结构化数据**，且在**特征工程后**效果更佳。
    """)

def display_xgboost_process(n_trees, learning_rate, max_depth=3, subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0):
    """
    显示XGBoost迭代学习过程
    """
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    
    # 生成更强的非线性数据
    np.random.seed(42)
    X = np.random.rand(300, 2) * 10  # 使用2个特征
    # 创建复杂的非线性关系
    y = 10 * np.sin(X[:, 0]) + 20 * np.cos(X[:, 1]) + \
        5 * np.sin(X[:, 0] * X[:, 1]) + np.random.normal(0, 3, size=X.shape[0])
    
    # 分割训练和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 显示数据分布
    st.markdown("### 数据集可视化")
    fig = plt.figure(figsize=(12, 5))
    
    # 3D可视化数据
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X_train[:, 0], X_train[:, 1], y_train, c='b', alpha=0.5, label='训练集')
    ax1.set_xlabel('特征1')
    ax1.set_ylabel('特征2')
    ax1.set_zlabel('目标值')
    ax1.set_title('训练数据的3D分布')
    
    # 训练集和测试集MSE可视化准备
    ax2 = fig.add_subplot(122)
    
    # 存储迭代过程中的预测和性能指标
    train_scores = []
    test_scores = []
    
    # 使用不同数量的树训练模型并记录性能
    for i in range(1, n_trees + 1):
        # 训练模型
        model = xgb.XGBRegressor(
            n_estimators=i,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            objective='reg:squarederror'
        )
        model.fit(X_train, y_train)
        
        # 预测并计算MSE
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_mse = np.mean((train_pred - y_train) ** 2)
        test_mse = np.mean((test_pred - y_test) ** 2)
        
        # 存储结果
        train_scores.append(train_mse)
        test_scores.append(test_mse)
    
    # 绘制MSE随树数量的变化
    ax2.plot(range(1, n_trees + 1), train_scores, 'b-', label='训练集MSE')
    ax2.plot(range(1, n_trees + 1), test_scores, 'r-', label='测试集MSE')
    ax2.set_xlabel('树的数量')
    ax2.set_ylabel('均方误差 (MSE)')
    ax2.set_title('MSE随树数量变化')
    ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 过拟合检测和最佳树数量
    best_n_trees = np.argmin(test_scores) + 1
    
    # 判断是否存在过拟合
    overfit_threshold = 5  # 当测试MSE连续5个点上升时判定为过拟合
    overfit_start = None
    for i in range(best_n_trees, len(test_scores) - overfit_threshold):
        if all(test_scores[i] < test_scores[j] for j in range(i+1, i+overfit_threshold+1)):
            overfit_start = i
            break
    
    st.markdown(f"""
    ### XGBoost性能分析
    
    - **最优树数量**: {best_n_trees} (测试集MSE最低: {min(test_scores):.2f})
    - **训练集最终MSE**: {train_scores[-1]:.2f}
    - **测试集最终MSE**: {test_scores[-1]:.2f}
    """)
    
    if overfit_start:
        st.warning(f"检测到从第 {overfit_start+1} 棵树开始可能存在过拟合现象！")
    elif train_scores[-1] < 0.3 * test_scores[-1]:
        st.warning(f"训练MSE显著低于测试MSE，可能存在过拟合！")
    
    # 可视化特征重要性
    final_model = xgb.XGBRegressor(
        n_estimators=n_trees,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda
    )
    final_model.fit(X_train, y_train)
    
    # 特征重要性图
    if X.shape[1] > 1:
        plt.figure(figsize=(10, 4))
        xgb.plot_importance(final_model)
        plt.title('XGBoost特征重要性')
        st.pyplot(plt)
    
    # 可视化预测效果
    st.markdown("### 预测效果可视化")
    
    # 创建网格数据用于可视化
    x1_range = np.linspace(min(X[:, 0]), max(X[:, 0]), 50)
    x2_range = np.linspace(min(X[:, 1]), max(X[:, 1]), 50)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    X_grid = np.column_stack((x1_grid.ravel(), x2_grid.ravel()))
    
    # 预测
    pred_grid = final_model.predict(X_grid).reshape(x1_grid.shape)
    
    # 计算真实值（用于比较）
    y_true_grid = (10 * np.sin(x1_grid) + 20 * np.cos(x2_grid) + 
                   5 * np.sin(x1_grid * x2_grid))
    
    # 绘制预测表面和真实数据点
    fig = plt.figure(figsize=(15, 6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(x1_grid, x2_grid, pred_grid, cmap='viridis', alpha=0.7)
    ax1.scatter(X_test[:, 0], X_test[:, 1], y_test, c='r', s=20, label='测试点')
    ax1.set_title('XGBoost预测表面')
    ax1.set_xlabel('特征1')
    ax1.set_ylabel('特征2')
    ax1.set_zlabel('预测值')
    
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(x1_grid, x2_grid, y_true_grid, cmap='plasma', alpha=0.7)
    ax2.scatter(X_test[:, 0], X_test[:, 1], y_test, c='r', s=20, label='测试点')
    ax2.set_title('真实关系表面')
    ax2.set_xlabel('特征1')
    ax2.set_ylabel('特征2')
    ax2.set_zlabel('真实值')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 添加解释性文字
    st.markdown(f"""
    ### 参数影响分析
    
    现在的模型配置:
    - **树的数量**: {n_trees}
    - **学习率**: {learning_rate}
    - **最大深度**: {max_depth}
    - **样本采样比例**: {subsample}
    - **特征采样比例**: {colsample_bytree}
    - **L2正则化系数**: {reg_lambda}
    
    #### 参数调整建议:
    
    1. **过拟合迹象**: 如果测试MSE开始上升但训练MSE继续下降，尝试:
       - 减少树的最大深度
       - 增加正则化系数
       - 降低学习率并增加树的数量
       
    2. **欠拟合迹象**: 如果训练和测试MSE都较高，尝试:
       - 增加树的最大深度
       - 增加树的数量
       - 减少正则化系数
    
    3. **最优平衡**: 理想情况下，训练MSE和测试MSE应该接近且较低，这表明模型既拟合了数据又有良好的泛化能力。
    """)

def show_xgboost_comparison():
    st.header("XGBoost算法流程与比较分析")
    
    # 添加XGBoost算法流程和性能比较
    st.subheader("XGBoost算法流程与模型比较")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### XGBoost算法流程
        
        1. **初始化**：使用一个常数值作为初始预测（如所有样本的平均值）
           $F_0(x) = \arg\min_c \sum_{i=1}^n l(y_i, c)$
        
        2. **对每棵树 t = 1, 2, ..., T**:
           
           a. 计算当前模型的梯度和海森矩阵:
              $g_i = \partial_{\hat{y}_i^{(t-1)}} l(y_i, \hat{y}_i^{(t-1)})$
              $h_i = \partial_{\hat{y}_i^{(t-1)}}^2 l(y_i, \hat{y}_i^{(t-1)})$
           
           b. 贪婪地生成树结构，对每个候选分裂计算增益:
              $Gain = \\frac{1}{2} \\left[ \\frac{G_L^2}{H_L+\\lambda} + \\frac{G_R^2}{H_R+\\lambda} - \\frac{G^2}{H+\\lambda} \\right] - \\gamma$
           
           c. 为每个叶子节点分配最优权重:
              $w_j = -\\frac{G_j}{H_j+\\lambda}$
           
           d. 将新树添加到模型中:
              $F_t(x) = F_{t-1}(x) + \\eta \\cdot f_t(x)$
        
        3. **输出最终模型**: $F_T(x) = \\sum_{t=0}^T f_t(x)$
        
        ### XGBoost特有技术
        
        1. **分裂点查找的近似算法**：
           - 对连续特征值进行分桶，减少搜索空间
           - 支持分布式加权分位数算法
        
        2. **稀疏感知算法**：
           - 自动处理缺失值
           - 为缺失值学习最佳方向
        
        3. **内存优化**：
           - 块压缩和列存储
           - 缓存感知预取
           - 核外计算支持
        """)
    
    with col2:
        st.markdown("""
        ### 与其他模型的性能比较
        
        | 特性 | XGBoost | 随机森林 | GBDT | LightGBM |
        |------|---------|----------|------|----------|
        | 速度 | 快 | 非常快（可并行） | 慢 | 非常快 |
        | 内存使用 | 中等 | 大 | 小 | 小 |
        | 准确性 | 高 | 中 | 中高 | 高 |
        | 过拟合抵抗力 | 高 | 非常高 | 中 | 中高 |
        | 可解释性 | 中 | 中 | 中 | 中 |
        | 大数据处理 | 好 | 好 | 较差 | 非常好 |
        | 缺失值处理 | 内置 | 需预处理 | 需预处理 | 内置 |
        | 类别变量处理 | 需编码 | 需编码 | 需编码 | 内置 |
        | 正则化 | 内置 | 无 | 有限 | 内置 |
        | 高维稀疏数据 | 适中 | 较差 | 适中 | 极好 |
        
        ### 适用场景选择
        
        - **XGBoost**: 中等规模数据，需要高精度
        - **随机森林**: 噪声大，防止过拟合是首要目标
        - **GBDT**: 传统非并行环境，小数据集
        - **LightGBM**: 非常大的数据集，需要极高速度
        
        ### 性能优势
        
        XGBoost在**Kaggle竞赛**中的表现突出：
        - 2015年有超过一半的冠军解决方案使用了XGBoost
        - 在大多数结构化数据问题中，它表现优于单一模型
        - 作为集成基础模型时尤其强大
        """)
    
    # 添加算法性能对比可视化
    st.subheader("集成学习算法性能对比")
    
    # 创建样本数据和指标
    methods = ['线性回归', '决策树', 'GBDT', '随机森林', 'XGBoost', 'LightGBM']
    small_data = [0.65, 0.72, 0.79, 0.81, 0.84, 0.83]  # 小数据集性能
    large_data = [0.64, 0.68, 0.75, 0.79, 0.83, 0.85]  # 大数据集性能
    training_speed = [1.0, 0.8, 0.2, 0.6, 0.5, 0.9]    # 训练速度（相对值，越高越快）
    inference_speed = [1.0, 0.9, 0.7, 0.6, 0.7, 0.8]   # 推理速度（相对值，越高越快）
    
    # 创建绘图区域
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('集成学习算法性能对比', fontsize=16)
    
    # 小数据集性能
    axs[0, 0].bar(methods, small_data, color=['gray', 'green', 'blue', 'orange', 'red', 'purple'])
    axs[0, 0].set_title('小数据集性能 (R²)')
    axs[0, 0].set_ylim([0.5, 0.9])
    axs[0, 0].set_ylabel('R²值')
    
    # 大数据集性能
    axs[0, 1].bar(methods, large_data, color=['gray', 'green', 'blue', 'orange', 'red', 'purple'])
    axs[0, 1].set_title('大数据集性能 (R²)')
    axs[0, 1].set_ylim([0.5, 0.9])
    axs[0, 1].set_ylabel('R²值')
    
    # 训练速度
    axs[1, 0].bar(methods, training_speed, color=['gray', 'green', 'blue', 'orange', 'red', 'purple'])
    axs[1, 0].set_title('训练速度 (相对值)')
    axs[1, 0].set_ylim([0, 1.1])
    axs[1, 0].set_ylabel('速度')
    
    # 推理速度
    axs[1, 1].bar(methods, inference_speed, color=['gray', 'green', 'blue', 'orange', 'red', 'purple'])
    axs[1, 1].set_title('推理速度 (相对值)')
    axs[1, 1].set_ylim([0, 1.1])
    axs[1, 1].set_ylabel('速度')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    st.pyplot(fig)
    
    # 添加雷达图比较
    st.subheader("算法多维性能对比")
    
    # 设置数据
    categories = ['准确性', '速度', '内存效率', '可扩展性', '易用性']
    
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合雷达图
    
    # XGBoost性能数据
    xgb_values = [0.9, 0.8, 0.7, 0.9, 0.8]
    xgb_values += xgb_values[:1]
    
    # 随机森林性能数据
    rf_values = [0.8, 0.9, 0.5, 0.7, 0.9]
    rf_values += rf_values[:1]
    
    # GBDT性能数据
    gbdt_values = [0.7, 0.5, 0.8, 0.6, 0.7]
    gbdt_values += gbdt_values[:1]
    
    # LightGBM性能数据
    lgb_values = [0.9, 0.9, 0.9, 0.9, 0.7]
    lgb_values += lgb_values[:1]
    
    # 创建雷达图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # 绘制各算法性能曲线
    ax.plot(angles, xgb_values, 'o-', linewidth=2, label='XGBoost', color='red')
    ax.fill(angles, xgb_values, alpha=0.1, color='red')
    
    ax.plot(angles, rf_values, 'o-', linewidth=2, label='随机森林', color='orange')
    ax.fill(angles, rf_values, alpha=0.1, color='orange')
    
    ax.plot(angles, gbdt_values, 'o-', linewidth=2, label='GBDT', color='blue')
    ax.fill(angles, gbdt_values, alpha=0.1, color='blue')
    
    ax.plot(angles, lgb_values, 'o-', linewidth=2, label='LightGBM', color='purple')
    ax.fill(angles, lgb_values, alpha=0.1, color='purple')
    
    # 设置雷达图属性
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.set_title('集成学习算法多维性能对比', size=15)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    st.pyplot(fig)
    
    st.markdown("""
    ### 性能对比分析
    
    根据上述图表，可以观察到：
    
    1. **准确性**：XGBoost和LightGBM在大多数场景下表现最佳，尤其是在结构化数据上
    
    2. **训练速度**：
       - 小数据集：随机森林最快，XGBoost次之
       - 大数据集：LightGBM表现最佳，XGBoost在大数据集上需要更多资源
    
    3. **内存使用**：LightGBM最高效，XGBoost次之，随机森林占用内存相对较多
    
    4. **可扩展性**：LightGBM和XGBoost支持分布式训练和并行计算，扩展性强
    
    5. **特点总结**：
       - XGBoost：全面且强大，是竞赛和企业级应用的首选
       - LightGBM：超大数据集和资源受限环境的最佳选择
       - 随机森林：易用且稳定，不易过拟合，适合作为基线模型
       - GBDT：原始算法实现，在小数据集上仍有应用价值
    """)
    
    # 添加XGBoost vs LightGBM对比
    st.subheader("XGBoost与LightGBM特点对比")
    
    comparison_data = {
        "特性": ["树生成策略", "分裂算法", "并行化方式", "内存使用", "速度", "精度", "支持平台"],
        "XGBoost": ["按层生长(level-wise)", "预排序 + 直方图", "特征并行", "较高", "快", "高", "全平台"],
        "LightGBM": ["按叶生长(leaf-wise)", "直方图优化", "数据并行+特征并行", "低", "非常快", "高", "全平台"]
    }
    
    st.table(pd.DataFrame(comparison_data).set_index("特性"))

def show_stacking():
    st.header("Stacking集成学习")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### Stacking的基本原理
        
        Stacking(堆叠集成)是一种将多个不同模型组合在一起的集成学习方法，通过"元学习器"(meta-learner)来整合基础模型的预测结果。
        
        ### 与其他集成方法的对比
        
        | 集成方法 | 基本思想 | 训练方式 | 权重确定 | 适用场景 |
        |---------|---------|----------|----------|---------|
        | Bagging | 并行训练多个相同类型模型 | 并行 | 平均/投票 | 高方差模型(如决策树) |
        | Boosting | 串行训练模型，关注难分样本 | 串行 | 加权 | 高偏差模型(如浅层树) |
        | Stacking | 使用元模型组合基础模型 | 分层 | 元模型学习 | 多种类型模型综合 |
        
        Stacking相比其他集成方法的最大特点是:
        
        1. **模型多样性**: 可以组合完全不同类型的模型(如线性模型、树模型、神经网络)
        2. **自动权重学习**: 元学习器自动学习最优的组合权重
        3. **表达能力**: 能捕捉不同模型在不同区域的优势
        """)
        
        st.subheader("Stacking工作流程")
        st.markdown("""
        1. **第一层：基础模型训练**
           - 训练多个不同的基础模型(例如RandomForest、XGBoost、LightGBM等)
           - 通常使用K折交叉验证生成"无泄漏"的预测
        
        2. **第二层：元模型训练**
           - 将基础模型的预测结果作为新特征
           - 训练元模型学习如何最佳组合这些预测
        
        3. **预测阶段**
           - 对新数据，先通过所有基础模型生成预测
           - 将这些预测输入到元模型中获得最终预测
        """)
    
    with col2:
        st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_stack_predictors_001.png", 
                caption="Stacking示意图（来源: scikit-learn）")
        
        st.markdown("""
        ### 元学习器选择
        
        常见的元学习器包括:
        
        - **线性模型**: 简单有效，不易过拟合
        - **树模型**: 能学习非线性组合关系
        - **简单神经网络**: 表达能力强但需要更多数据
        
        理想的元学习器应该:
        - 足够简单，避免过拟合
        - 能处理特征之间的相关性
        - 计算效率高
        """)
    
    # 交互式Stacking演示
    st.subheader("交互式Stacking过程可视化")
    
    # 数据集选择
    dataset = st.selectbox(
        "选择数据集",
        ["回归问题 (波士顿房价)", "分类问题 (鸢尾花)"]
    )
    
    # 基础模型选择
    st.markdown("#### 选择基础模型")
    col1, col2, col3 = st.columns(3)
    with col1:
        use_lr = st.checkbox("线性模型", value=True)
    with col2:
        use_rf = st.checkbox("随机森林", value=True)
    with col3:
        use_gb = st.checkbox("梯度提升树", value=True)
    
    # 元模型选择
    meta_model = st.selectbox(
        "选择元模型",
        ["线性回归/分类", "随机森林", "XGBoost"]
    )
    
    # 验证方式选择
    cv_folds = st.slider("交叉验证折数", 2, 10, 5)
    
    if st.button("运行Stacking演示"):
        show_stacking_demo(dataset, use_lr, use_rf, use_gb, meta_model, cv_folds)
    
    # Stacking优缺点
    st.subheader("Stacking的优缺点")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### 优点
        
        1. **性能提升**: 通常比单一模型表现更好
        2. **自动权重分配**: 元模型能学到最优的组合方式
        3. **领域灵活性**: 适用于几乎所有机器学习问题
        4. **减少过拟合**: 元学习器有助于平滑预测结果
        5. **降低方差**: 多个模型组合减小随机性影响
        """)
    
    with col2:
        st.markdown("""
        #### 缺点
        
        1. **计算复杂度**: 需要训练多个模型，计算成本高
        2. **实现复杂**: 交叉验证生成偏置训练特征较复杂
        3. **调参难度**: 需要为多个模型调参
        4. **解释性降低**: 最终模型解释性不如单一模型
        5. **边际收益递减**: 随模型数量增加，性能提升变小
        """)
    
    # Stacking实际应用
    st.subheader("Stacking的应用场景")
    st.markdown("""
    Stacking在许多领域都有成功应用:
    
    1. **竞赛**: 几乎所有数据科学竞赛的顶级解决方案都使用了Stacking
    2. **金融预测**: 结合不同模型预测股价、风险评估
    3. **医疗诊断**: 组合多种模型提高疾病诊断准确率
    4. **推荐系统**: 整合不同推荐算法的结果
    5. **计算机视觉**: 组合CNN、LSTM等模型进行图像分类
    
    Stacking最适合:
    - 有足够计算资源的场景
    - 对精度要求极高的任务
    - 可用多种不同类型模型的问题
    """)
    
    # 实现代码示例
    st.subheader("Python实现示例")
    st.code("""
# 使用scikit-learn实现Stacking
from sklearn.ensemble import StackingRegressor  # 或 StackingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# 定义基础模型
base_models = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('svr', SVR(kernel='rbf'))
]

# 定义元模型
meta_model = LinearRegression()

# 创建Stacking模型
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5  # 5折交叉验证
)

# 训练和预测
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
stacking_model.fit(X_train, y_train)
predictions = stacking_model.predict(X_test)
    """, language="python")

def show_stacking_demo(dataset, use_lr, use_rf, use_gb, meta_model, cv_folds):
    """展示Stacking过程的交互式演示"""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.metrics import mean_squared_error, accuracy_score
    import numpy as np
    
    st.markdown("### Stacking流程演示")
    
    # 加载数据集
    if dataset == "回归问题 (波士顿房价)":
        # 波士顿房价数据集已在scikit-learn中弃用，这里使用模拟数据
        np.random.seed(42)
        n_samples = 500
        n_features = 10
        X = np.random.randn(n_samples, n_features)
        # 创建一些非线性关系
        y = 5 + np.sum(X[:, :3], axis=1) + 3 * (X[:, 3] ** 2) + np.random.randn(n_samples) * 2
        st.info("注意: 使用的是模拟的房价数据集，而非真实的波士顿房价数据")
        task_type = "regression"
        metric_name = "均方根误差 (RMSE)"
        lower_is_better = True
    else:
        X, y = load_iris(return_X_y=True)
        task_type = "classification"
        metric_name = "准确率"
        lower_is_better = False
    
    # 创建进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 可视化准备
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 定义模型
    models = []
    model_names = []
    
    status_text.text("准备模型...")
    progress_bar.progress(10)
    
    if task_type == "regression":
        if use_lr:
            models.append(LinearRegression())
            model_names.append("线性回归")
        if use_rf:
            models.append(RandomForestRegressor(n_estimators=100, random_state=42))
            model_names.append("随机森林")
        if use_gb:
            models.append(GradientBoostingRegressor(n_estimators=100, random_state=42))
            model_names.append("梯度提升树")
            
        if meta_model == "线性回归/分类":
            final_model = LinearRegression()
            meta_name = "线性回归"
        elif meta_model == "随机森林":
            final_model = RandomForestRegressor(n_estimators=50, random_state=42)
            meta_name = "随机森林"
        else:  # XGBoost
            try:
                import xgboost as xgb
                final_model = xgb.XGBRegressor(n_estimators=50, random_state=42)
                meta_name = "XGBoost"
            except ImportError:
                final_model = GradientBoostingRegressor(n_estimators=50, random_state=42)
                meta_name = "梯度提升树 (XGBoost未安装)"
    else:  # 分类
        if use_lr:
            models.append(LogisticRegression(max_iter=1000, random_state=42))
            model_names.append("逻辑回归")
        if use_rf:
            models.append(RandomForestClassifier(n_estimators=100, random_state=42))
            model_names.append("随机森林")
        if use_gb:
            models.append(GradientBoostingClassifier(n_estimators=100, random_state=42))
            model_names.append("梯度提升树")
            
        if meta_model == "线性回归/分类":
            final_model = LogisticRegression(max_iter=1000, random_state=42)
            meta_name = "逻辑回归"
        elif meta_model == "随机森林":
            final_model = RandomForestClassifier(n_estimators=50, random_state=42)
            meta_name = "随机森林"
        else:  # XGBoost
            try:
                import xgboost as xgb
                final_model = xgb.XGBClassifier(n_estimators=50, random_state=42)
                meta_name = "XGBoost"
            except ImportError:
                final_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
                meta_name = "梯度提升树 (XGBoost未安装)"
    
    if len(models) == 0:
        st.error("请至少选择一个基础模型!")
        return
    
    # 生成交叉验证预测
    status_text.text("第一层: 使用交叉验证为基础模型生成预测...")
    progress_bar.progress(30)
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    meta_features = np.zeros((X.shape[0], len(models)))
    
    # 基础模型性能
    base_performances = []
    
    for i, model in enumerate(models):
        # 使用交叉验证生成预测
        if task_type == "regression":
            preds = cross_val_predict(model, X, y, cv=kf)
            score = np.sqrt(mean_squared_error(y, preds))
        else:
            preds = cross_val_predict(model, X, y, cv=kf)
            score = accuracy_score(y, preds)
        
        meta_features[:, i] = preds
        base_performances.append(score)
        
        # 更新进度
        progress_percent = 30 + (i + 1) * 20 / len(models)
        progress_bar.progress(int(progress_percent))
        status_text.text(f"训练基础模型: {model_names[i]} 完成")
    
    # 第二层: 元模型训练
    status_text.text("第二层: 训练元模型...")
    progress_bar.progress(70)
    
    # 使用交叉验证评估堆叠模型
    if task_type == "regression":
        meta_preds = cross_val_predict(final_model, meta_features, y, cv=kf)
        stack_score = np.sqrt(mean_squared_error(y, meta_preds))
    else:
        meta_preds = cross_val_predict(final_model, meta_features, y, cv=kf)
        stack_score = accuracy_score(y, meta_preds)
    
    # 展示完整结果
    progress_bar.progress(100)
    status_text.text("演示完成!")
    
    # 绘制性能对比条形图
    all_models = model_names + ["Stacking集成"]
    all_scores = base_performances + [stack_score]
    
    # 调整颜色
    colors = ['blue'] * len(model_names) + ['red']
    
    # 绘制性能对比
    axes[0].bar(all_models, all_scores, color=colors)
    axes[0].set_title(f'模型性能对比 ({metric_name})')
    axes[0].set_ylabel(metric_name)
    
    # 在正确的位置添加标签
    for i, v in enumerate(all_scores):
        axes[0].text(i, v * 1.05, f'{v:.3f}', ha='center')
    
    if lower_is_better:
        best_idx = np.argmin(all_scores)
        worst_idx = np.argmax(all_scores)
    else:
        best_idx = np.argmax(all_scores)
        worst_idx = np.argmin(all_scores)
    
    axes[0].get_children()[best_idx].set_color('green')
    
    # 绘制预测分布散点图
    if task_type == "regression":
        # 随机选择部分样本以避免图形过于拥挤
        np.random.seed(42)
        sample_idx = np.random.choice(len(y), min(100, len(y)), replace=False)
        
        for i, model_name in enumerate(model_names):
            axes[1].scatter(y[sample_idx], meta_features[sample_idx, i], 
                         alpha=0.5, label=model_name)
        
        axes[1].scatter(y[sample_idx], meta_preds[sample_idx], 
                      color='red', marker='X', s=100, label='Stacking集成')
        
        axes[1].plot([min(y), max(y)], [min(y), max(y)], 'k--', lw=2)
        axes[1].set_xlabel('实际值')
        axes[1].set_ylabel('预测值')
        axes[1].set_title('预测值与实际值对比')
        axes[1].legend()
    else:
        # 为分类问题创建简单的混淆矩阵热图
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(y, meta_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
        axes[1].set_title('Stacking模型混淆矩阵')
        axes[1].set_xlabel('预测类别')
        axes[1].set_ylabel('实际类别')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 性能提升分析
    if task_type == "regression" and lower_is_better:
        base_avg = np.mean(base_performances)
        improvement = (base_avg - stack_score) / base_avg * 100
        best_base = min(base_performances)
        best_improvement = (best_base - stack_score) / best_base * 100
        
        if stack_score < best_base:
            st.success(f"Stacking相比所有单一模型平均提升了 {improvement:.2f}%，相比最佳单一模型提升了 {best_improvement:.2f}%")
        elif stack_score < base_avg:
            st.info(f"Stacking相比单一模型平均提升了 {improvement:.2f}%，但没有超过最佳单一模型")
        else:
            st.warning("在这个例子中，Stacking没有提升性能，可能是因为数据集太小或基础模型相似度高")
    elif task_type == "classification" and not lower_is_better:
        base_avg = np.mean(base_performances)
        improvement = (stack_score - base_avg) / base_avg * 100
        best_base = max(base_performances)
        best_improvement = (stack_score - best_base) / best_base * 100
        
        if stack_score > best_base:
            st.success(f"Stacking相比所有单一模型平均提升了 {improvement:.2f}%，相比最佳单一模型提升了 {best_improvement:.2f}%")
        elif stack_score > base_avg:
            st.info(f"Stacking相比单一模型平均提升了 {improvement:.2f}%，但没有超过最佳单一模型")
        else:
            st.warning("在这个例子中，Stacking没有提升性能，可能是因为数据集太小或基础模型相似度高")

def show_evaluation_metrics():
    st.header("回归模型评估指标选择")
    
    st.markdown("""
    ### 常用回归评估指标
    
    评估回归模型性能时，选择合适的指标至关重要，不同指标反映模型性能的不同方面。
    """)
    
    # 评估指标解释
    metrics = {
        "均方误差(MSE)": {
            "公式": r"$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$",
            "特点": "对离群值敏感，强调大误差",
            "适用场景": "预测值不能有大偏差的场景",
            "数值示例": "若实际值为10，预测值为15，贡献为(15-10)²=25"
        },
        "均方根误差(RMSE)": {
            "公式": r"$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$",
            "特点": "MSE的平方根，与因变量同单位",
            "适用场景": "需要与因变量相同单位的场景",
            "数值示例": "若MSE=25，则RMSE=5"
        },
        "平均绝对误差(MAE)": {
            "公式": r"$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$",
            "特点": "对离群值不敏感，所有误差权重相同",
            "适用场景": "存在异常值且不希望过分惩罚的场景",
            "数值示例": "若实际值为10，预测值为15，贡献为|15-10|=5"
        },
        "平均绝对百分比误差(MAPE)": {
            "公式": r"$\text{MAPE} = \frac{1}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100\%$",
            "特点": "衡量相对误差，与数据尺度无关",
            "适用场景": "需要比较不同量级预测的准确性",
            "数值示例": "若实际值为10，预测值为15，贡献为|15-10|/10×100%=50%"
        },
        "决定系数(R²)": {
            "公式": r"$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$",
            "特点": "衡量模型解释的方差比例，范围通常在[0,1]",
            "适用场景": "需要评估模型相对于平均值的改进程度",
            "数值示例": "R²=0.8表示模型解释了80%的因变量方差"
        }
    }
    
    # 交互式选择指标
    selected_metric = st.selectbox("选择评估指标查看详情", list(metrics.keys()))
    
    # 显示所选指标的详情
    st.subheader(selected_metric)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**数学公式:**")
        st.latex(metrics[selected_metric]["公式"].replace(r"$", ""))
        
        st.markdown("**特点:**")
        st.markdown(metrics[selected_metric]["特点"])
    
    with col2:
        st.markdown("**适用场景:**")
        st.markdown(metrics[selected_metric]["适用场景"])
        
        st.markdown("**数值示例:**")
        st.markdown(metrics[selected_metric]["数值示例"])
    
    # 指标可视化比较
    st.subheader("评估指标对比可视化")
    
    # 生成示例数据
    np.random.seed(42)
    y_true = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    
    # 创建几个不同的预测结果
    y_pred1 = y_true + np.random.normal(0, 5, size=len(y_true))  # 小误差
    y_pred2 = y_true + np.random.normal(0, 10, size=len(y_true))  # 中等误差
    y_pred3 = y_true.copy()
    y_pred3[7] = 180  # 加入一个离群值
    
    # 计算各种评估指标
    def calculate_metrics(y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "R²": r2
        }
    
    metrics1 = calculate_metrics(y_true, y_pred1)
    metrics2 = calculate_metrics(y_true, y_pred2)
    metrics3 = calculate_metrics(y_true, y_pred3)
    
    # 创建比较表格
    df_compare = pd.DataFrame({
        "小误差模型": metrics1,
        "中等误差模型": metrics2,
        "含离群值模型": metrics3
    })
    
    st.table(df_compare.round(2))
    
    # 可视化不同模型的预测与实际值
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(y_true, 'bo-', label='实际值')
    axes[0].plot(y_pred1, 'ro--', label='预测值')
    axes[0].set_title('小误差模型')
    axes[0].legend()
    
    axes[1].plot(y_true, 'bo-', label='实际值')
    axes[1].plot(y_pred2, 'ro--', label='预测值')
    axes[1].set_title('中等误差模型')
    axes[1].legend()
    
    axes[2].plot(y_true, 'bo-', label='实际值')
    axes[2].plot(y_pred3, 'ro--', label='预测值')
    axes[2].set_title('含离群值模型')
    axes[2].legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 指标选择建议
    st.subheader("指标选择建议")
    st.markdown("""
    * **选择原则**: 根据业务需求选择最合适的指标
    * **多指标结合**: 通常使用多个指标结合评估
    * **考虑误差影响**: 评估预测偏差的严重程度与业务影响
    
    **具体建议**:
    * 对离群值敏感的问题选择MAE而非MSE/RMSE
    * 需要评估相对误差时使用MAPE
    * 比较不同模型时，R²是一个直观的指标
    * 对于直接与业务KPI相关的预测，使用与业务指标直接相关的评估指标
    """)
    
    # 交互问题
    st.subheader("思考问题")
    q1 = st.multiselect(
        "在以下场景中，哪些评估指标更合适？",
        ["MSE", "RMSE", "MAE", "MAPE", "R²"],
        default=[]
    )
    
    scenario = st.selectbox(
        "选择场景",
        ["房价预测", "销售额预测", "温度预测", "股价预测"]
    )
    
    recommendations = {
        "房价预测": ["RMSE", "MAE", "R²"],
        "销售额预测": ["MAPE", "MAE", "R²"],
        "温度预测": ["MAE", "RMSE"],
        "股价预测": ["MAPE", "R²"]
    }
    
    if q1:
        if set(q1).issubset(set(recommendations[scenario])) and len(q1) > 0:
            st.success(f"好选择！{q1}确实适合{scenario}场景。")
        else:
            st.warning(f"对于{scenario}，通常推荐使用{', '.join(recommendations[scenario])}。")
            st.markdown(f"原因：")
            if scenario == "房价预测":
                st.markdown("- RMSE：与目标变量单位相同，容易解释")
                st.markdown("- MAE：减少异常房价的影响")
                st.markdown("- R²：评估模型解释房价变化的能力")
            elif scenario == "销售额预测":
                st.markdown("- MAPE：销售额预测通常关注相对误差")
                st.markdown("- MAE：对不同量级的产品销售额预测都公平")
                st.markdown("- R²：了解模型解释销售额方差的比例")
            elif scenario == "温度预测":
                st.markdown("- MAE：温度预测通常关注平均绝对误差，对异常值不敏感")
                st.markdown("- RMSE：均方根误差，对较大误差更敏感，在乎预测的精确度")
            elif scenario == "股价预测":
                st.markdown("- MAPE：股价预测更关注相对误差，百分比误差更直观")
                st.markdown("- R²：评估模型对股价波动趋势的解释能力")