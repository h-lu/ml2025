import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from scipy.spatial.distance import cdist
import pandas as pd
import time

from utils.app_utils import (
    create_custom_header, 
    create_info_box, 
    render_latex, 
    plot_step_by_step_controls,
    euclidean_distance_calculator,
    create_expander
)
from utils.data_generator import generate_blob_data, generate_custom_data, generate_anisotropic_data
from utils.visualization import (
    plot_clusters, 
    plot_kmeans_steps, 
    plot_elbow_method,
    plot_kmeans_centroid_sensitivity
)

def show_kmeans():
    """显示K-means聚类页面"""
    create_custom_header("K-means 聚类：简单高效的划分方法", "最常用的聚类算法之一", "🎯")
    
    # 创建标签页
    tab1, tab2, tab3, tab4 = st.tabs([
        "算法核心思想与步骤", 
        "互动环节：手动K-means", 
        "关键参数K的选择", 
        "K-means的优缺点"
    ])
    
    with tab1:
        show_kmeans_algorithm()
    
    with tab2:
        show_kmeans_interactive()
    
    with tab3:
        show_kmeans_k_selection()
    
    with tab4:
        show_kmeans_pros_cons()

def show_kmeans_algorithm():
    """显示K-means算法核心思想和步骤"""
    st.subheader("算法核心思想与步骤")
    
    # 目标
    st.markdown("""
    **目标：** 将数据划分为 K 个簇，使得每个数据点都属于离其最近的簇的质心（均值中心），
    同时最小化簇内平方和 (Within-Cluster Sum of Squares, WCSS)。
    """)
    
    # 数学表达
    st.markdown("**数学表达式：**")
    render_latex(r"J = \sum_{j=1}^{K} \sum_{i \in C_j} ||x_i - \mu_j||^2")
    
    st.markdown("""
    其中：
    * $C_j$ 是第 $j$ 个簇
    * $\mu_j$ 是簇 $C_j$ 的质心
    * $x_i$ 是簇 $C_j$ 中的数据点
    """)
    
    # 算法步骤
    st.markdown("### 详细步骤：")
    
    # 步骤选择器
    steps = [
        "1. 初始化 (Initialization)", 
        "2. 分配 (Assignment)", 
        "3. 更新 (Update)", 
        "4. 迭代 (Iteration)"
    ]
    
    selected_step = st.selectbox("选择要查看的步骤：", steps)
    
    if selected_step == steps[0]:
        st.markdown("""
        **初始化 (Initialization):** **随机**选择 K 个数据点作为初始质心 ($\mu_1, \mu_2, ..., \mu_K$)。
        
        *思考：随机初始化可能导致什么问题？还有其他初始化方法吗？*
        """)
        
        with st.expander("初始化方法"):
            st.markdown("""
            1. **随机选择：** 从数据点中随机选择K个点作为初始质心。简单但可能导致不良的聚类结果。
            
            2. **K-means++：** 一种改进的初始化方法，核心思想是使初始质心尽可能分散。
               * 第一个质心随机选择
               * 后续质心选择时，距离现有质心越远的点被选为新质心的概率越大
               * 这种方法能显著提高K-means的聚类质量和收敛速度
               
            3. **分区平均：** 将数据空间分成K个区域，使用每个区域的均值作为初始质心。
            
            4. **层次聚类结果：** 先进行层次聚类，然后使用结果作为K-means的初始质心。
            """)
    
    elif selected_step == steps[1]:
        st.markdown("""
        **分配 (Assignment):** 对于数据集中的**每一个**数据点 \(x_i\)，计算它到**所有** K 个质心 \(\mu_j\) 的距离 (通常使用欧氏距离)，
        并将其分配给距离**最近**的质心所代表的簇 \(C_j\)。
        
        **欧氏距离公式:**
        """)
        render_latex(r"d(x, \mu) = \sqrt{\sum_{d=1}^{D}(x_d - \mu_d)^2}")
        
        st.markdown("""
        其中 \(D\) 是数据维度。
        
        *思考：为什么常用欧氏距离？它有什么几何意义？还有其他距离度量吗？*
        """)
        
        with st.expander("距离度量方法"):
            st.markdown("""
            1. **欧氏距离 (Euclidean Distance)：** 直线距离，最常用的度量方法，适合当特征空间中的各个方向同等重要时。
               * 公式：$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$
            
            2. **曼哈顿距离 (Manhattan Distance)：** 也称城市街区距离，沿坐标轴方向的距离和。
               * 公式：$d(x, y) = \sum_{i=1}^{n} |x_i - y_i|$
               * 适合在网格状空间中移动的情况
            
            3. **余弦相似度 (Cosine Similarity)：** 衡量两个向量方向的相似性，忽略幅度差异。
               * 公式：$\cos(θ) = \\frac{x \\cdot y}{||x|| ||y||}$
               * 适合文本分析等需要考虑方向而非幅度的场景
            
            4. **马氏距离 (Mahalanobis Distance)：** 考虑特征间相关性的距离度量。
               * 适合处理特征间存在相关性的数据集
            """)
        
        # 提供欧氏距离计算器
        if st.checkbox("尝试欧氏距离计算器"):
            euclidean_distance_calculator()
    
    elif selected_step == steps[2]:
        st.markdown("""
        **更新 (Update):** 对于**每一个**簇 \(C_j\)，重新计算其质心 \(\mu_j\)，即该簇中所有数据点的**均值**。
        """)
        render_latex(r"\mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i")
        
        st.markdown("""
        其中 \(|C_j|\) 是簇 \(C_j\) 中的数据点数量。
        """)
    
    elif selected_step == steps[3]:
        st.markdown("""
        **迭代 (Iteration):** 重复步骤 2 (分配) 和步骤 3 (更新)，直到满足停止条件。
        
        **停止条件：**
        * 质心位置不再发生显著变化（例如，移动距离小于某个阈值）。
        * 数据点的簇分配不再改变。
        * 达到预设的最大迭代次数。
        """)
    
    # 可视化演示
    st.subheader("K-means 迭代过程可视化演示")
    
    # 生成一些数据用于演示
    X, y = generate_blob_data(n_samples=150, n_centers=3, random_state=42)
    
    # 用户可调参数
    k = st.slider("选择簇的数量 (K)：", 2, 5, 3)
    iterations = st.slider("显示迭代次数：", 1, 5, 2)
    
    # 初始化K-means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    
    # 绘制步骤图
    initial_centroids = kmeans.cluster_centers_
    if k == 3:
        # 使用预设的初始质心使结果更好看
        initial_centroids = np.array([
            [-1.0, 0.0],
            [1.0, 1.0],
            [0.0, -1.0]
        ])
    
    figs = plot_kmeans_steps(X, initial_centroids, iterations)
    
    # 显示图形
    st.write("初始状态")
    st.pyplot(figs[0])
    
    for i in range(iterations):
        st.write(f"迭代 {i+1}: 分配阶段")
        st.pyplot(figs[2*i+1])
        
        st.write(f"迭代 {i+1}: 更新阶段")
        st.pyplot(figs[2*i+2])
    
    # 最终聚类结果
    st.subheader("K-means 最终聚类结果")
    
    final_kmeans = KMeans(n_clusters=k, init=initial_centroids, n_init=1)
    final_kmeans.fit(X)
    
    final_labels = final_kmeans.labels_
    final_centroids = final_kmeans.cluster_centers_
    
    fig = plot_clusters(X, final_labels, final_centroids, title="最终聚类结果")
    st.pyplot(fig)

def show_kmeans_interactive():
    """手动K-means迭代互动环节"""
    st.subheader("互动环节：手动 K-means 模拟")
    
    st.markdown("""
    在这个互动环节中，我们将手动执行K-means聚类算法的步骤，
    以加深对算法原理的理解。我们将使用一个简单的2D数据集，包含6个点。
    """)
    
    # 准备数据
    X = generate_custom_data()
    point_names = ['A', 'B', 'C', 'D', 'E', 'F']
    
    # 显示数据点
    df = pd.DataFrame(X, columns=['x', 'y'])
    df['点'] = point_names
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("数据点坐标：")
        st.table(df)
    
    with col2:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(X[:, 0], X[:, 1], s=100)
        for i, name in enumerate(point_names):
            ax.annotate(name, (X[i, 0], X[i, 1]), xytext=(5, 5), textcoords='offset points')
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        st.pyplot(fig)
    
    st.markdown("---")
    
    # 选择初始质心
    st.markdown("### 第1步：选择初始质心")
    
    init_options = [
        "选择A和D作为初始质心",
        "选择B和F作为初始质心"
    ]
    
    init_choice = st.radio("请选择初始质心：", init_options)
    
    if init_choice == init_options[0]:
        centroids = np.array([X[0], X[3]])  # A和D
        centroid_names = ['A', 'D']
    else:
        centroids = np.array([X[1], X[5]])  # B和F
        centroid_names = ['B', 'F']
    
    # 绘制初始质心
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], s=100)
    
    for i, name in enumerate(point_names):
        ax.annotate(name, (X[i, 0], X[i, 1]), xytext=(5, 5), textcoords='offset points')
    
    ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c=['r', 'g'], marker='X', label='质心')
    
    for i, name in enumerate(centroid_names):
        ax.annotate(f'质心 {name}', (centroids[i, 0], centroids[i, 1]), 
                   xytext=(10, 10), textcoords='offset points', 
                   bbox=dict(boxstyle="round,pad=0.3", fc='yellow', alpha=0.7))
    
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('初始质心')
    
    st.pyplot(fig)
    
    st.markdown("---")
    
    # 第一次迭代 - 分配步骤
    st.markdown("### 第2步：计算每个点到质心的距离并分配")
    
    # 计算距离
    distances = cdist(X, centroids)
    labels = np.argmin(distances, axis=1)
    
    # 创建距离表格
    dist_df = pd.DataFrame()
    dist_df['点'] = point_names
    dist_df[f'到{centroid_names[0]}的距离'] = distances[:, 0].round(2)
    dist_df[f'到{centroid_names[1]}的距离'] = distances[:, 1].round(2)
    dist_df['分配给'] = [centroid_names[l] for l in labels]
    
    st.write("距离计算和分配：")
    st.table(dist_df)
    
    # 绘制分配结果
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['r', 'g']
    for i in range(2):  # 2个簇
        cluster_points = X[labels == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=100, c=colors[i], alpha=0.6, label=f'簇 {i+1}')
    
    for i, name in enumerate(point_names):
        ax.annotate(name, (X[i, 0], X[i, 1]), xytext=(5, 5), textcoords='offset points')
    
    ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c=colors, marker='X')
    
    for i, name in enumerate(centroid_names):
        ax.annotate(f'质心 {name}', (centroids[i, 0], centroids[i, 1]), 
                   xytext=(10, 10), textcoords='offset points', 
                   bbox=dict(boxstyle="round,pad=0.3", fc='yellow', alpha=0.7))
    
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('迭代1: 分配结果')
    
    st.pyplot(fig)
    
    st.markdown("---")
    
    # 第一次迭代 - 更新步骤
    st.markdown("### 第3步：更新质心位置")
    
    # 更新质心
    new_centroids = np.array([
        X[labels == 0].mean(axis=0) if np.sum(labels == 0) > 0 else centroids[0],
        X[labels == 1].mean(axis=0) if np.sum(labels == 1) > 0 else centroids[1]
    ])
    
    # 创建更新前后质心表格
    centroid_df = pd.DataFrame()
    centroid_df['质心'] = [f'质心1 ({centroid_names[0]})', f'质心2 ({centroid_names[1]})']
    centroid_df['更新前坐标'] = [f'({centroids[0,0]:.2f}, {centroids[0,1]:.2f})', 
                              f'({centroids[1,0]:.2f}, {centroids[1,1]:.2f})']
    centroid_df['更新后坐标'] = [f'({new_centroids[0,0]:.2f}, {new_centroids[0,1]:.2f})', 
                              f'({new_centroids[1,0]:.2f}, {new_centroids[1,1]:.2f})']
    
    st.write("质心更新：")
    st.table(centroid_df)
    
    # 绘制更新结果
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i in range(2):  # 2个簇
        cluster_points = X[labels == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=100, c=colors[i], alpha=0.6, label=f'簇 {i+1}')
    
    for i, name in enumerate(point_names):
        ax.annotate(name, (X[i, 0], X[i, 1]), xytext=(5, 5), textcoords='offset points')
    
    # 绘制旧质心（虚线轮廓）
    ax.scatter(centroids[:, 0], centroids[:, 1], s=200, facecolors='none', edgecolors=colors, 
              linewidth=2, linestyle='--', alpha=0.5)
    
    # 绘制新质心
    ax.scatter(new_centroids[:, 0], new_centroids[:, 1], s=200, c=colors, marker='X', label='新质心')
    
    # 绘制质心移动箭头
    for i in range(2):
        ax.arrow(centroids[i, 0], centroids[i, 1],
                new_centroids[i, 0] - centroids[i, 0],
                new_centroids[i, 1] - centroids[i, 1],
                head_width=0.2, head_length=0.3, fc=colors[i], ec=colors[i])
    
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('迭代1: 更新质心')
    
    st.pyplot(fig)
    
    # 更新质心
    centroids = new_centroids
    
    st.markdown("---")
    
    # 第二次迭代 - 分配步骤
    st.markdown("### 第4步：第二次迭代 - 分配")
    
    # 计算距离
    distances = cdist(X, centroids)
    new_labels = np.argmin(distances, axis=1)
    
    # 创建距离表格
    dist_df = pd.DataFrame()
    dist_df['点'] = point_names
    dist_df['到质心1的距离'] = distances[:, 0].round(2)
    dist_df['到质心2的距离'] = distances[:, 1].round(2)
    dist_df['分配给'] = [f'质心{l+1}' for l in new_labels]
    dist_df['与上一次分配相比'] = ['相同' if new_labels[i] == labels[i] else '变化' for i in range(len(X))]
    
    st.write("距离计算和分配：")
    st.table(dist_df)
    
    # 绘制分配结果
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i in range(2):  # 2个簇
        cluster_points = X[new_labels == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=100, c=colors[i], alpha=0.6, label=f'簇 {i+1}')
    
    for i, name in enumerate(point_names):
        ax.annotate(name, (X[i, 0], X[i, 1]), xytext=(5, 5), textcoords='offset points')
    
    ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c=colors, marker='X')
    
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('迭代2: 分配结果')
    
    st.pyplot(fig)
    
    # 检查是否分配变化
    if np.array_equal(new_labels, labels):
        st.success("分配没有变化，算法已收敛！")
    else:
        st.info("分配发生了变化，需要继续迭代。")
        labels = new_labels
    
    st.markdown("---")
    
    # 讨论
    st.subheader("讨论")
    
    st.markdown("""
    通过手动执行K-means算法的步骤，你应该能够更清楚地理解算法的工作原理。现在，思考以下问题：
    
    1. 两种不同的初始化方式最终得到的聚类结果是否相同？这说明了K-means的什么特点？
    
    2. 如何缓解K-means对初始质心敏感的问题？（提示：多次运行取最优、更智能的初始化方法）
    
    3. 在实际应用中，如何确定簇的数量K？（稍后我们会讨论肘部法则等方法）
    """)

def show_kmeans_k_selection():
    """关键参数K的选择"""
    st.subheader("关键参数 K 的选择：肘部法则 (Elbow Method)")
    
    st.markdown("""
    K-means 的一个关键挑战是选择合适的簇数量 K。如果我们知道数据中应该有多少个簇，
    那么选择 K 会很简单。但在大多数实际应用中，我们并不知道最佳的 K 值。
    
    **肘部法则**是一种简单而常用的确定 K 值的方法。
    """)
    
    # 肘部法则思想
    st.markdown("""
    **肘部法则思想:**
    
    1. 尝试不同的 K 值（例如，从 1 到 10）。
    2. 对于每个 K 值，运行 K-means 算法，并计算簇内平方和 (WCSS)。
    3. 绘制 K 值与 WCSS 的关系图。
    4. 观察图像，寻找曲线下降速率趋于平缓的"肘部"对应的 K 值。
    
    这个"肘部"通常被认为是 WCSS 下降带来的收益（解释了更多方差）与增加簇数量带来的复杂性之间的较好平衡点。
    """)
    
    # 交互式肘部法则演示
    st.markdown("### 交互式肘部法则演示")
    
    # 用户选择数据类型
    data_type = st.radio(
        "选择数据类型：",
        ["标准球状数据 (3个簇)", "不同大小和密度的簇 (3个簇)"],
        horizontal=True
    )
    
    if data_type == "标准球状数据 (3个簇)":
        X, _ = generate_blob_data(n_samples=300, n_centers=3, random_state=42)
    else:
        X, _ = generate_anisotropic_data(n_samples=300, n_centers=3, random_state=42)
    
    # 用户选择最大K值
    max_k = st.slider("最大K值：", 2, 15, 10)
    
    # 绘制肘部法则图
    fig = plot_elbow_method(X, max_k)
    st.pyplot(fig)
    
    # 用户选择K值
    selected_k = st.slider("选择K值：", 2, max_k, 3)
    
    # 显示所选K值的聚类结果
    kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    
    fig = plot_clusters(X, labels, centroids, title=f"K={selected_k}的聚类结果")
    st.pyplot(fig)
    
    # 局限性讨论
    st.markdown("""
    **肘部法则的局限性：**
    
    * "肘部"有时不明显，难以确定。
    * 对于某些数据集（如复杂形状或重叠的簇），WCSS可能不是最佳指标。
    * 该方法仅提供参考，通常需要结合领域知识和其他评估指标。
    
    **其他确定K值的方法：**
    
    * **轮廓系数 (Silhouette Coefficient):** 衡量簇内相似度与簇间差异性。
    * **间隙统计量 (Gap Statistic):** 比较观察的WCSS与随机参考分布。
    * **贝叶斯信息准则 (BIC) 或阿卡克信息准则 (AIC):** 平衡模型复杂度和拟合程度。
    * **X-means:** K-means的扩展，自动确定簇数量。
    * **领域知识:** 有时基于业务需求或领域知识选择K更合适。
    """)

def show_kmeans_pros_cons():
    """K-means的优缺点"""
    st.subheader("K-means 的优缺点总结")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 优点")
        st.markdown("""
        * **简单直观：** 算法逻辑清晰，易于理解和实现。
        
        * **高效：** 对于大规模数据集，计算速度相对较快，时间复杂度接近线性 $O(N × K × D × I)$，其中 N 是样本数，K 是簇数，D 是维度，I 是迭代次数。
        
        * **适用性强：** 在许多领域有广泛应用，如客户细分、图像压缩、特征学习等。
        
        * **可扩展：** 容易扩展为在线学习版本（Mini-batch K-means）。
        """)
    
    with col2:
        st.markdown("### 缺点")
        st.markdown("""
        * **对初始质心敏感：** 不同的初始点可能导致不同的聚类结果，甚至可能陷入局部最优。
        
        * **需预先指定 K 值：** K 值的选择对结果影响很大，且没有绝对最优的方法确定 K。
        
        * **对噪声和异常值敏感：** 异常值会对均值计算产生较大影响，可能导致质心偏移。
        
        * **假设簇为凸状/球状：** 对于非凸形状（如环状、月牙状）或大小/密度差异很大的簇，效果不佳。
        
        * **仅适用于数值型数据：** 标准 K-means 基于均值和欧氏距离，难以直接处理类别型数据 (需要转换)。
        """)
    
    # 演示K-means对初始质心的敏感性
    st.subheader("K-means 对初始质心的敏感性演示")
    
    st.markdown("""
    K-means算法的聚类结果对初始质心的选择非常敏感，尤其是在处理形状复杂的数据时。
    下面的演示使用同样的数据集但不同的初始质心，观察最终的聚类结果有何不同。
    """)
    
    # 使用plot_kmeans_centroid_sensitivity函数绘制，它会自动生成适合演示的数据
    fig = plot_kmeans_centroid_sensitivity(None, K=3, n_init=3)
    st.pyplot(fig)
    
    st.markdown("""
    从上图可以清楚地看到，**不同的初始质心会导致显著不同的聚类结果：**
    
    * **簇的形状和大小不同：** 可以看到不同初始化条件下，簇的划分边界明显不同
    * **惯性(Inertia)不同：** 惯性是衡量簇内距离平方和的指标，值越小表示聚类效果越好
    * **极易陷入局部最优：** 特别是对于环形或复杂形状的数据，K-means很容易找到次优解
    
    为了缓解这个问题，通常的做法是：
    
    1. **多次运行K-means**，使用不同的随机初始化，选择簇内平方和最小的结果。
       这就是scikit-learn中KMeans的`n_init`参数的作用，默认为10。
    
    2. **使用更智能的初始化方法**，如K-means++，它使初始质心尽可能分散，这是scikit-learn的默认初始化方法。
    
    3. **使用更鲁棒的聚类算法**，如K-medoids，它使用实际数据点作为中心而不是均值。
    
    4. **对于特定形状的数据**，考虑使用其他更适合的聚类算法，如基于密度的DBSCAN或谱聚类。
    """)
    
    # 演示K-means的形状限制
    st.subheader("K-means 对簇形状的限制")
    
    data_shape = st.selectbox(
        "选择数据形状：",
        ["环形数据", "月牙形数据"]
    )
    
    if data_shape == "环形数据":
        from sklearn.datasets import make_circles
        X, y = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)
    else:
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=300, noise=0.05, random_state=42)
    
    # 原始数据
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
    ax.set_title("原始数据（真实簇）")
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)
    
    # K-means结果
    k = 2  # 真实簇数
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    kmeans_centroids = kmeans.cluster_centers_
    
    fig = plot_clusters(X, kmeans_labels, kmeans_centroids, title="K-means聚类结果")
    st.pyplot(fig)
    
    st.markdown("""
    如上所示，K-means在处理非凸形状的簇时表现不佳。这是因为K-means假设簇是凸的，
    并且使用欧氏距离和均值作为簇中心。
    
    对于这类数据，基于密度的聚类算法（如DBSCAN）或谱聚类（Spectral Clustering）
    通常能获得更好的结果。
    """) 