import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import pandas as pd

from utils.app_utils import create_custom_header, create_info_box, render_latex, create_expander
from utils.data_generator import generate_blob_data
from utils.visualization import plot_clusters, plot_dendrogram, plot_different_linkages

def show_hierarchical():
    """显示层次聚类页面"""
    create_custom_header("层次聚类：构建簇的层级结构", "提供数据的层次化表示", "🌳")
    
    # 创建标签页
    tab1, tab2, tab3, tab4 = st.tabs([
        "核心思想", 
        "步骤与树状图", 
        "Linkage方法", 
        "优缺点"
    ])
    
    with tab1:
        show_hierarchical_intro()
    
    with tab2:
        show_hierarchical_steps()
    
    with tab3:
        show_linkage_methods()
    
    with tab4:
        show_hierarchical_pros_cons()

def show_hierarchical_intro():
    """层次聚类的核心思想"""
    st.subheader("核心思想：合并或分裂")
    
    st.markdown("""
    层次聚类是一类通过创建簇的层次结构来进行聚类的算法。与K-means不同，层次聚类不需要预先指定簇的数量，
    它提供了数据的多层次分组视图，可以直观地看到数据的嵌套结构。
    """)
    
    # 两种主要策略
    st.markdown("### 两种主要策略")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **凝聚型 (Agglomerative) - 自底向上 (Bottom-up):**
        
        1. 开始时，每个数据点自成一簇。
        2. 在每一步，合并**最相似**（距离最近）的两个簇。
        3. 重复此过程，直到所有点合并成一个大簇。
        
        *本节课重点讲解凝聚型。*
        """)
        
        # 添加简单的自底向上示意图
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # 模拟5个数据点
        points = np.array([
            [1, 1],
            [1.5, 1.5],
            [5, 5],
            [5.5, 5.5],
            [3, 3]
        ])
        
        # 绘制数据点
        ax.scatter(points[:, 0], points[:, 1], s=100, c='blue')
        
        # 添加文本标签
        for i, (x, y) in enumerate(points):
            ax.text(x+0.1, y+0.1, f'点{i+1}', fontsize=10)
        
        # 画出合并顺序
        ax.plot([points[0, 0], points[1, 0]], [points[0, 1], points[1, 1]], 'r--', lw=2)
        ax.text(1.25, 1.25, '第1步', color='red')
        
        ax.plot([points[2, 0], points[3, 0]], [points[2, 1], points[3, 1]], 'g--', lw=2)
        ax.text(5.25, 5.25, '第2步', color='green')
        
        ax.plot([1.25, 5.25], [1.25, 5.25], 'b--', lw=2)
        ax.text(3, 3.2, '第4步', color='blue')
        
        ax.plot([3, 3], [3, 3], 'k--', lw=2)
        ax.text(3.1, 3, '第3步', color='black')
        
        ax.set_title('凝聚型层次聚类示意图')
        st.pyplot(fig)
    
    with col2:
        st.markdown("""
        **分裂型 (Divisive) - 自顶向下 (Top-down):**
        
        1. 开始时，所有数据点属于同一个簇。
        2. 在每一步，将一个簇分裂成两个**最不相似**（距离最远）的子簇。
        3. 重复此过程，直到每个点自成一簇或达到某个停止条件（如指定的簇数量）。
        
        *分裂型计算复杂度通常更高。*
        """)
        
        # 添加简单的自顶向下示意图
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # 使用相同的数据点
        ax.scatter(points[:, 0], points[:, 1], s=100, c='blue')
        
        # 添加文本标签
        for i, (x, y) in enumerate(points):
            ax.text(x+0.1, y+0.1, f'点{i+1}', fontsize=10)
        
        # 画出分裂顺序
        # 第1步：分成左右两组
        ax.plot([3, 3], [0, 6], 'r--', lw=2)
        ax.text(3, 4, '第1步', color='red')
        
        # 第2步：右边组进一步分裂
        ax.plot([4.5, 4.5], [4, 6], 'g--', lw=2)
        ax.text(4.5, 4.5, '第2步', color='green')
        
        # 第3步：左边组进一步分裂
        ax.plot([2, 2], [0, 3.5], 'b--', lw=2)
        ax.text(2, 2, '第3步', color='blue')
        
        ax.set_title('分裂型层次聚类示意图')
        st.pyplot(fig)
    
    # 讨论两种方法的应用场景
    st.markdown("""
    **两种方法的对比：**
    
    1. **计算复杂度：** 凝聚型通常计算复杂度为O(n³)，分裂型可能更高。
    
    2. **应用场景：** 
       - 凝聚型适合于簇数量较多的情况，特别是数据点较少时。
       - 分裂型在期望大型簇结构，并且对细节不太关注时可能更有用。
    
    3. **实现难度：** 凝聚型更容易实现，也是更常见的层次聚类方法。
    
    在实际应用中，凝聚型层次聚类使用更为广泛，本课程将重点介绍凝聚型方法。
    """)

def show_hierarchical_steps():
    """层次聚类的步骤与树状图"""
    st.subheader("凝聚型层次聚类的步骤与树状图 (Dendrogram)")
    
    # 详细步骤
    st.markdown("### 详细步骤 (凝聚型):")
    
    steps = [
        "1. 初始化", 
        "2. 查找最近簇", 
        "3. 合并", 
        "4. 更新距离矩阵", 
        "5. 迭代"
    ]
    
    selected_step = st.selectbox("选择要查看的步骤：", steps)
    
    if selected_step == steps[0]:
        st.markdown("""
        **初始化：** 将 N 个数据点各自视为一个簇，共 N 个簇。计算所有点对之间的距离，形成距离矩阵。
        
        例如，对于4个数据点，其距离矩阵可能如下：
        """)
        
        # 示例距离矩阵
        dist_matrix = np.array([
            [0.0, 1.5, 4.2, 5.1],
            [1.5, 0.0, 3.8, 4.5],
            [4.2, 3.8, 0.0, 2.0],
            [5.1, 4.5, 2.0, 0.0]
        ])
        
        df = pd.DataFrame(
            dist_matrix,
            index=['点1', '点2', '点3', '点4'],
            columns=['点1', '点2', '点3', '点4']
        )
        
        st.table(df)
    
    elif selected_step == steps[1]:
        st.markdown("""
        **查找最近簇：** 在距离矩阵中找到距离最小的两个簇 C_i 和 C_j。
        
        在初始阶段，簇就是单个数据点。如果距离矩阵中点1和点2的距离最小，那么这两个点形成的簇将被合并。
        """)
    
    elif selected_step == steps[2]:
        st.markdown("""
        **合并：** 将簇 C_i 和 C_j 合并成一个新的簇 C_new。
        
        这一步会减少簇的总数，例如从N个簇变为N-1个簇。
        """)
    
    elif selected_step == steps[3]:
        st.markdown("""
        **更新距离矩阵：** 从矩阵中移除 C_i 和 C_j 的行和列，添加新簇 C_new 的行和列。
        
        计算 C_new 与其他现有簇 C_k 之间的距离。**关键在于如何定义簇间距离，即 Linkage 方法**。
        
        不同的linkage方法会导致不同的聚类结果，例如：
        - 单linkage：新簇与另一簇的距离 = 合并前两个簇中与另一簇距离较小的那个
        - 全linkage：新簇与另一簇的距离 = 合并前两个簇中与另一簇距离较大的那个
        - 平均linkage：新簇与另一簇的距离 = 合并前两个簇与另一簇距离的平均值
        """)
    
    elif selected_step == steps[4]:
        st.markdown("""
        **迭代：** 重复步骤2-4，直到只剩下一个簇。
        
        在这个过程中，我们可以记录每一次合并操作以及合并时的距离，这些信息可以用来构建树状图（Dendrogram）。
        """)
    
    # 树状图介绍
    st.markdown("### 树状图 (Dendrogram)")
    
    st.markdown("""
    树状图是可视化层次聚类过程的重要工具，它展示了数据点是如何逐步合并成更大的簇的。
    """)
    
    # 生成一些数据用于演示
    X, y = generate_blob_data(n_samples=20, n_centers=3, random_state=42)
    
    # 绘制原始数据点
    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=100, alpha=0.8, cmap='viridis')
    ax.set_title("原始数据点")
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)
    
    # 计算层次聚类的linkage矩阵
    Z = linkage(X, 'ward')
    
    # 绘制树状图
    fig = plot_dendrogram(Z, title="层次聚类树状图示例")
    st.pyplot(fig)
    
    # 树状图解读
    st.markdown("""
    **树状图解读:**
    
    * **叶节点：** 代表原始数据点。
    * **纵轴：** 通常表示簇合并时的距离或不相似度。合并发生的高度越高，表示合并的簇之间距离越远。
    * **横轴：** 代表数据点或簇。
    * **合并点：** 水平线连接的两个或多个分支表示这些簇在该纵轴高度（距离）被合并。
    
    **如何确定簇数量:** 在树状图上选择一个"切割高度"（水平线），与该水平线相交的竖线数量即为最终得到的簇数量。
    """)
    
    # 交互式选择切割高度
    cut_height = st.slider(
        "选择切割高度（确定簇数量）:", 
        float(Z[:, 2].min()), 
        float(Z[:, 2].max()), 
        float(Z[:, 2].mean()),
        step=0.1
    )
    
    # 绘制带有切割线的树状图
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(
        Z,
        ax=ax,
        orientation='top',
        color_threshold=cut_height
    )
    ax.axhline(y=cut_height, c='r', lw=1, linestyle='--')
    ax.set_title("带切割线的树状图")
    st.pyplot(fig)
    
    # 计算切割后的簇数量
    clusters = np.unique(fcluster(Z, cut_height, criterion='distance')).size
    st.write(f"**在高度 {cut_height:.2f} 切割得到 {clusters} 个簇**")
    
    # 显示聚类结果
    labels = fcluster(Z, cut_height, criterion='distance') - 1  # 簇标签从0开始
    
    fig = plot_clusters(X, labels, title=f"切割高度 {cut_height:.2f} 的聚类结果")
    st.pyplot(fig)

def show_linkage_methods():
    """Linkage方法及其影响"""
    st.subheader("Linkage 方法：定义簇间距离")
    
    st.markdown("""
    Linkage 方法定义了如何计算两个簇之间的距离。不同的 Linkage 方法可能导致非常不同的聚类结果。
    """)
    
    # 核心问题
    st.markdown("""
    **核心问题：** 如何衡量两个簇（而不是两个点）之间的距离？
    
    当两个簇各自包含多个点时，我们需要一种方法来定义这两个簇之间的距离。
    """)
    
    # 常用Linkage方法
    st.markdown("### 常用 Linkage 方法")
    
    linkage_methods = {
        "single": {
            "name": "单 Linkage (Single Linkage / Minimum Linkage)",
            "definition": "两个簇之间的距离 = 两个簇中**最近**的两个点之间的距离。",
            "formula": r"D(C_i, C_j) = \min_{x \in C_i, y \in C_j} d(x, y)",
            "characteristics": "倾向于产生链状的、细长的簇，对噪声敏感。容易受到 \"链式效应\" 影响。"
        },
        "complete": {
            "name": "全 Linkage (Complete Linkage / Maximum Linkage)",
            "definition": "两个簇之间的距离 = 两个簇中**最远**的两个点之间的距离。",
            "formula": r"D(C_i, C_j) = \max_{x \in C_i, y \in C_j} d(x, y)",
            "characteristics": "倾向于产生紧凑的、球状的簇，对异常值没有单 Linkage 那么敏感。"
        },
        "average": {
            "name": "平均 Linkage (Average Linkage)",
            "definition": "两个簇之间的距离 = 两个簇中所有点对距离的**平均值**。",
            "formula": r"D(C_i, C_j) = \frac{1}{|C_i||C_j|} \sum_{x \in C_i} \sum_{y \in C_j} d(x, y)",
            "characteristics": "效果介于 Single 和 Complete Linkage 之间，较为常用。"
        },
        "ward": {
            "name": "Ward's Linkage",
            "definition": "合并两个簇，使得合并后所有簇的**总簇内平方和增量最小**。旨在最小化方差。",
            "formula": "复杂，基于合并前后的总平方和增量",
            "characteristics": "倾向于产生大小相似、方差较小的球状簇，对噪声敏感。常与欧氏距离配合使用。"
        }
    }
    
    # 用户选择Linkage方法
    method = st.selectbox(
        "选择Linkage方法:",
        ["single", "complete", "average", "ward"],
        format_func=lambda x: linkage_methods[x]["name"]
    )
    
    # 显示选中的Linkage方法信息
    method_info = linkage_methods[method]
    
    st.markdown(f"**{method_info['name']}:**")
    st.markdown(f"**定义：** {method_info['definition']}")
    
    st.markdown("**数学表达式：**")
    render_latex(method_info["formula"])
    
    st.markdown(f"**特点：** {method_info['characteristics']}")
    
    # 示意图 - 简单展示不同linkage如何计算距离
    st.markdown("### 不同Linkage距离计算示意图")
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    
    # 生成两组点
    np.random.seed(42)
    cluster1 = np.random.randn(5, 2) * 0.5 + np.array([-2, 0])
    cluster2 = np.random.randn(5, 2) * 0.5 + np.array([2, 0])
    
    titles = ["Single Linkage", "Complete Linkage", "Average Linkage"]
    
    for i, title in enumerate(titles):
        axs[i].scatter(cluster1[:, 0], cluster1[:, 1], s=80, c='red', label='簇1')
        axs[i].scatter(cluster2[:, 0], cluster2[:, 1], s=80, c='blue', label='簇2')
        
        if i == 0:  # Single Linkage
            # 找到距离最短的两点
            min_dist = float('inf')
            min_points = None
            
            for p1 in cluster1:
                for p2 in cluster2:
                    dist = np.linalg.norm(p1 - p2)
                    if dist < min_dist:
                        min_dist = dist
                        min_points = (p1, p2)
            
            # 画出最短距离
            axs[i].plot([min_points[0][0], min_points[1][0]], 
                        [min_points[0][1], min_points[1][1]], 
                        'k--', lw=2)
            axs[i].text((min_points[0][0] + min_points[1][0]) / 2, 
                        (min_points[0][1] + min_points[1][1]) / 2 + 0.2, 
                        f"dist = {min_dist:.2f}")
            
        elif i == 1:  # Complete Linkage
            # 找到距离最长的两点
            max_dist = 0
            max_points = None
            
            for p1 in cluster1:
                for p2 in cluster2:
                    dist = np.linalg.norm(p1 - p2)
                    if dist > max_dist:
                        max_dist = dist
                        max_points = (p1, p2)
            
            # 画出最长距离
            axs[i].plot([max_points[0][0], max_points[1][0]], 
                        [max_points[0][1], max_points[1][1]], 
                        'k--', lw=2)
            axs[i].text((max_points[0][0] + max_points[1][0]) / 2, 
                        (max_points[0][1] + max_points[1][1]) / 2 + 0.2, 
                        f"dist = {max_dist:.2f}")
            
        else:  # Average Linkage
            # 计算所有点对距离的和
            total_dist = 0
            count = 0
            
            for p1 in cluster1:
                for p2 in cluster2:
                    dist = np.linalg.norm(p1 - p2)
                    total_dist += dist
                    count += 1
            
            avg_dist = total_dist / count
            
            # 画出中心点连线
            c1_center = np.mean(cluster1, axis=0)
            c2_center = np.mean(cluster2, axis=0)
            
            axs[i].plot([c1_center[0], c2_center[0]], 
                        [c1_center[1], c2_center[1]], 
                        'k--', lw=2)
            axs[i].text((c1_center[0] + c2_center[0]) / 2, 
                        (c1_center[1] + c2_center[1]) / 2 + 0.2, 
                        f"avg dist = {avg_dist:.2f}")
        
        axs[i].set_title(title)
        axs[i].legend()
        axs[i].set_aspect('equal')
        axs[i].grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 比较不同Linkage方法在真实数据上的效果
    st.markdown("### 不同Linkage方法在真实数据上的效果比较")
    
    # 生成数据
    X, y = generate_blob_data(n_samples=100, n_centers=3, random_state=42)
    
    # 用户选择要比较的Linkage方法
    selected_methods = st.multiselect(
        "选择要比较的Linkage方法:",
        ["single", "complete", "average", "ward"],
        default=["single", "complete", "average", "ward"],
        format_func=lambda x: linkage_methods[x]["name"]
    )
    
    if selected_methods:
        # 绘制不同linkage方法的树状图
        fig = plot_different_linkages(X, selected_methods)
        st.pyplot(fig)
        
        # 绘制不同linkage方法的聚类结果
        n_clusters = st.slider("簇数量:", 2, 6, 3)
        
        fig, axs = plt.subplots(1, len(selected_methods), figsize=(15, 4))
        
        if len(selected_methods) == 1:
            axs = [axs]  # 确保axs是列表，即使只有一个subplot
        
        for i, method in enumerate(selected_methods):
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters, 
                linkage=method
            )
            labels = clustering.fit_predict(X)
            
            axs[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8)
            axs[i].set_title(f"{linkage_methods[method]['name']}")
            axs[i].set_aspect('equal')
            axs[i].grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 添加观察和讨论
        st.markdown("""
        **观察和讨论:**
        
        * **单Linkage (Single):** 容易产生"链式效应"，形成不均衡的簇，某些簇可能非常大，而其他簇很小。
        
        * **全Linkage (Complete):** 倾向于产生大小相似的簇，适合当簇的大小应该相对均衡时使用。
        
        * **平均Linkage (Average):** 是单Linkage和全Linkage的折中方案，通常产生较合理的结果。
        
        * **Ward Linkage:** 试图最小化簇内方差，倾向于产生大小相似、形状紧凑的簇。
        
        **什么时候选择哪种Linkage方法？**
        
        * 如果你希望发现细长或不规则形状的簇，可以考虑单Linkage。
        * 如果你希望簇的大小相对均衡，可以考虑全Linkage或Ward Linkage。
        * 对于大多数一般用途，平均Linkage通常是一个很好的默认选择。
        * 如果你关注簇内的紧凑性和方差最小化，Ward Linkage可能是最佳选择。
        """)

def show_hierarchical_pros_cons():
    """层次聚类的优缺点"""
    st.subheader("层次聚类的优缺点总结")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 优点")
        st.markdown("""
        * **无需预先指定簇数量：** 树状图提供了对不同簇数量划分的可视化，可以根据需求选择切割点。
        
        * **提供层次结构：** 树状图本身揭示了数据点之间的层次关系，有助于理解数据的嵌套结构。
        
        * **对簇形状假设较少：** 相较于K-means，对簇的形状（特别是Single/Average Linkage）没那么严格的假设。
        
        * **结果唯一确定：** 给定Linkage方法和距离度量，算法结果是确定的，不像K-means那样受初始化影响。
        
        * **适用于多种数据类型：** 只要能定义合适的距离度量，层次聚类可以应用于各种数据类型。
        """)
    
    with col2:
        st.markdown("### 缺点")
        st.markdown("""
        * **计算复杂度高：** 凝聚型算法的时间复杂度通常为O(n³)或O(n²log n)（使用优化的数据结构），空间复杂度为O(n²)（存储距离矩阵），难以处理非常大规模的数据集。
        
        * **合并/分裂不可撤销：** 一旦一个合并（或分裂）发生，后续步骤无法撤销，早期错误的合并可能影响最终结果。
        
        * **对距离度量和Linkage方法敏感：** 选择不同的度量和Linkage会显著影响结果。
        
        * **难以解释大型树状图：** 当数据点非常多时，树状图可能变得非常复杂，难以解读。
        
        * **存储需求大：** 需要存储n×n的距离矩阵，对于大型数据集可能消耗大量内存。
        """)
    
    # 计算复杂度对比
    st.subheader("层次聚类 vs. K-means 计算复杂度")
    
    complexity_data = {
        "算法": ["K-means", "层次聚类 (凝聚型)"],
        "时间复杂度": ["O(n × K × d × i)", "O(n³)"],
        "空间复杂度": ["O(n + K)", "O(n²)"],
        "适用数据规模": ["大型数据集", "中小型数据集"],
        "注": [
            "n=样本数, K=簇数, d=维度, i=迭代次数", 
            "优化实现可达到O(n²log n)"
        ]
    }
    
    complexity_df = pd.DataFrame(complexity_data)
    st.table(complexity_df)
    
    # 可扩展性讨论
    st.markdown("""
    ### 提高层次聚类可扩展性的方法
    
    对于大型数据集，标准层次聚类算法的计算复杂度可能成为限制因素。以下是一些可以提高算法可扩展性的方法：
    
    1. **BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies):** 
       一种为大型数据集设计的层次聚类算法，通过构建CF树（Clustering Feature Tree）来减少内存需求和计算量。
    
    2. **采样方法：** 
       对大型数据集进行采样，在较小的样本上执行层次聚类，然后将剩余点分配到最近的簇。
    
    3. **CURE (Clustering Using REpresentatives):** 
       使用多个代表点表示一个簇，而不是单个中心点，既能处理非球形簇，又能提高效率。
    
    4. **并行化实现：** 
       利用现代多核处理器和分布式计算平台实现并行化的层次聚类算法。
    """)
    
    # 应用场景
    st.subheader("层次聚类的典型应用场景")
    
    applications = {
        "领域": [
            "生物信息学", 
            "社交网络分析", 
            "文档聚类", 
            "客户细分", 
            "图像分割"
        ],
        "应用": [
            "构建基因或蛋白质的进化树", 
            "发现社区结构和层次关系", 
            "创建文档的主题层次结构", 
            "识别客户的多层次分组", 
            "图像中对象的多尺度分割"
        ],
        "为什么选择层次聚类": [
            "自然表示生物进化的层次关系", 
            "无需预先知道社区数量，可展示嵌套社区", 
            "主题通常有层次关系，如主题-子主题", 
            "客户群体可能有多层次结构", 
            "物体可能包含多层次的组件和子组件"
        ]
    }
    
    applications_df = pd.DataFrame(applications)
    st.table(applications_df)

# 从scipy.cluster.hierarchy导入fcluster函数
from scipy.cluster.hierarchy import fcluster 