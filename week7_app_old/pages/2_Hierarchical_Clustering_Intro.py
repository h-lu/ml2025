import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from utils.font_manager import set_chinese_font

st.set_page_config(
    page_title="层次聚类介绍",
    page_icon="🌲",
    layout="wide"
)

st.title("层次聚类基本原理")

st.markdown("""
层次聚类是一种不需要预先指定簇数量的聚类方法，它会构建一个嵌套的簇的层次结构。与K-Means不同，
层次聚类可以揭示数据内部的层次关系，并通过树状图直观地展示这种结构。
""")

# 生成凝聚式层次聚类示意图
def create_agglomerative_example():
    set_chinese_font()  # 设置中文字体
    
    # 创建示例数据点
    np.random.seed(42)
    X = np.random.rand(7, 2) * 10
    
    # 设置绘图样式
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    # 初始状态：每个点是一个簇
    axes[0].scatter(X[:, 0], X[:, 1], s=100, c=np.arange(7), cmap='tab10')
    for i, (x, y) in enumerate(X):
        axes[0].text(x+0.1, y+0.1, f'点{i+1}', fontsize=12)
    axes[0].set_title('步骤1: 每个点各自为一簇')
    
    # 第一次合并
    axes[1].scatter(X[[0,1,2,3,4,6], 0], X[[0,1,2,3,4,6], 1], s=100, c=[0,1,2,3,4,6], cmap='tab10')
    axes[1].scatter(X[5, 0], X[5, 1], s=100, c='purple')
    axes[1].plot([X[4, 0], X[5, 0]], [X[4, 1], X[5, 1]], 'k--')
    for i, (x, y) in enumerate(X):
        if i != 5:
            axes[1].text(x+0.1, y+0.1, f'点{i+1}', fontsize=12)
    axes[1].text(X[5, 0]+0.1, X[5, 1]+0.1, f'点6 (已合并到簇5)', fontsize=12)
    axes[1].set_title('步骤2: 合并最近的两个点')
    
    # 第二次合并
    axes[2].scatter(X[[0,2,3,4,6], 0], X[[0,2,3,4,6], 1], s=100, c=[0,2,3,4,6], cmap='tab10')
    axes[2].scatter(X[1, 0], X[1, 1], s=100, c='green')
    axes[2].scatter(X[5, 0], X[5, 1], s=100, c='purple')
    axes[2].plot([X[0, 0], X[1, 0]], [X[0, 1], X[1, 1]], 'k--')
    axes[2].plot([X[4, 0], X[5, 0]], [X[4, 1], X[5, 1]], 'k--')
    axes[2].set_title('步骤3: 继续合并最近的簇')
    
    # 第三次合并
    axes[3].scatter(X[[0,2,4,6], 0], X[[0,2,4,6], 1], s=100, c=[0,2,4,6], cmap='tab10')
    axes[3].scatter(X[1, 0], X[1, 1], s=100, c='green')
    axes[3].scatter(X[3, 0], X[3, 1], s=100, c='yellow')
    axes[3].scatter(X[5, 0], X[5, 1], s=100, c='purple')
    axes[3].plot([X[0, 0], X[1, 0]], [X[0, 1], X[1, 1]], 'k--')
    axes[3].plot([X[2, 0], X[3, 0]], [X[2, 1], X[3, 1]], 'k--')
    axes[3].plot([X[4, 0], X[5, 0]], [X[4, 1], X[5, 1]], 'k--')
    axes[3].set_title('步骤4: 继续合并')
    
    # 最终状态
    axes[4].scatter(X[[0,4], 0], X[[0,4], 1], s=100, c=[0,4], cmap='tab10')
    axes[4].scatter(X[1, 0], X[1, 1], s=100, c='green')
    axes[4].scatter(X[2, 0], X[2, 1], s=100, c='red')
    axes[4].scatter(X[3, 0], X[3, 1], s=100, c='yellow')
    axes[4].scatter(X[5, 0], X[5, 1], s=100, c='purple')
    axes[4].scatter(X[6, 0], X[6, 1], s=100, c='brown')
    axes[4].plot([X[0, 0], X[1, 0]], [X[0, 1], X[1, 1]], 'k--')
    axes[4].plot([X[2, 0], X[3, 0]], [X[2, 1], X[3, 1]], 'k--')
    axes[4].plot([X[4, 0], X[5, 0]], [X[4, 1], X[5, 1]], 'k--')
    axes[4].plot([X[4, 0], X[6, 0]], [X[4, 1], X[6, 1]], 'k--')
    axes[4].set_title('步骤5: 合并过程继续')
    
    # 最终一个簇
    axes[5].scatter(X[:, 0], X[:, 1], s=100, c='blue')
    for i in range(6):
        axes[5].plot([X[i, 0], X[i+1, 0]], [X[i, 1], X[i+1, 1]], 'k--')
    axes[5].set_title('最终: 所有点合并为一个簇')
    
    # 调整所有子图
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.grid(True)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    return fig

# 两种主要方式
st.header("两种主要方式")
col1, col2 = st.columns(2)

with col1:
    st.subheader("凝聚式 (Agglomerative)")
    st.markdown("""
    * **自底向上**的方法
    * 开始时每个数据点自成一簇
    * 逐步合并最相似的簇
    * 直到所有点合并为一个簇
    * **最常用**的方式
    """)
    
    # 使用本地生成的图代替网络图片
    agglomerative_fig = create_agglomerative_example()
    st.pyplot(agglomerative_fig)
    st.caption("凝聚式层次聚类示意图 - 展示了点逐步合并的过程")

with col2:
    st.subheader("分裂式 (Divisive)")
    st.markdown("""
    * **自顶向下**的方法
    * 开始时所有数据点在一个簇
    * 逐步分裂最不相似的簇
    * 直到每个点自成一簇
    * 计算复杂度较高，不太常用
    """)

# 凝聚式层次聚类步骤
st.header("凝聚式层次聚类步骤")
st.markdown("""
1. 将每个数据点视为一个单独的簇
2. 计算所有簇之间的距离（或相似度）
3. 合并距离最近（最相似）的两个簇
4. 重新计算新合并簇与其他簇之间的距离
5. 重复步骤3和4，直到所有数据点合并为一个簇
""")

# 关键概念
st.header("关键概念")

# 距离度量
st.subheader("距离度量 (Distance Metric)")
st.markdown("""
用于计算数据点或簇之间的距离，常用的距离度量包括：

* **欧氏距离 (Euclidean distance)**: 最常用的距离度量，适合连续型数据
* **曼哈顿距离 (Manhattan distance)**: 也称为城市街区距离，适合网格状数据
* **余弦相似度 (Cosine similarity)**: 适合高维稀疏数据，如文本
""")

# 连接标准
st.subheader("连接标准 (Linkage Criteria)")
st.markdown("""
定义如何计算簇之间的距离，不同的连接标准会产生不同的聚类结果：
""")

linkage_col1, linkage_col2 = st.columns(2)

with linkage_col1:
    st.markdown("""
    **Ward连接 (`linkage='ward'`)**
    * 最小化簇内方差的增量
    * 通常效果较好，倾向于产生大小相似的簇
    * 只能与欧氏距离一起使用
    
    **Average Linkage (`linkage='average'`)**
    * 计算两个簇中所有点对之间距离的平均值
    * 较为平衡，对异常值不太敏感
    """)

with linkage_col2:
    st.markdown("""
    **Complete Linkage (`linkage='complete'`)**
    * 计算两个簇中所有点对之间距离的最大值
    * 倾向于产生紧凑的球状簇

    **Single Linkage (`linkage='single'`)**
    * 计算两个簇中所有点对之间距离的最小值
    * 对异常值敏感，可能产生链状效应
    * 适合识别非凸形状的簇
    """)

# 使用本地生成的连接标准比较图
def create_linkage_comparison():
    set_chinese_font()  # 设置中文字体
    
    # 创建示例数据 - 两个簇
    np.random.seed(42)
    cluster1 = np.random.randn(10, 2) * 0.5 + np.array([2, 2])
    cluster2 = np.random.randn(10, 2) * 0.5 + np.array([6, 6])
    X = np.vstack([cluster1, cluster2])
    
    # 绘制不同连接标准
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    titles = ['Single连接 (最小距离)', 'Complete连接 (最大距离)', 
             'Average连接 (平均距离)', 'Ward连接 (最小方差增量)']
    
    for i, (title, ax) in enumerate(zip(titles, axes)):
        ax.scatter(X[:10, 0], X[:10, 1], color='blue', s=100, label='簇1')
        ax.scatter(X[10:, 0], X[10:, 1], color='red', s=100, label='簇2')
        
        # 绘制连接线
        if i == 0:  # Single - 连接最近的两点
            min_dist_idx1, min_dist_idx2 = 0, 0
            min_dist = float('inf')
            for a in range(10):
                for b in range(10, 20):
                    dist = np.linalg.norm(X[a] - X[b])
                    if dist < min_dist:
                        min_dist = dist
                        min_dist_idx1, min_dist_idx2 = a, b
            ax.plot([X[min_dist_idx1, 0], X[min_dist_idx2, 0]], 
                    [X[min_dist_idx1, 1], X[min_dist_idx2, 1]], 'k--', lw=2)
            ax.scatter([X[min_dist_idx1, 0], X[min_dist_idx2, 0]], 
                      [X[min_dist_idx1, 1], X[min_dist_idx2, 1]], 
                      color='green', s=150, zorder=10)
            
        elif i == 1:  # Complete - 连接最远的两点
            max_dist_idx1, max_dist_idx2 = 0, 0
            max_dist = 0
            for a in range(10):
                for b in range(10, 20):
                    dist = np.linalg.norm(X[a] - X[b])
                    if dist > max_dist:
                        max_dist = dist
                        max_dist_idx1, max_dist_idx2 = a, b
            ax.plot([X[max_dist_idx1, 0], X[max_dist_idx2, 0]], 
                    [X[max_dist_idx1, 1], X[max_dist_idx2, 1]], 'k--', lw=2)
            ax.scatter([X[max_dist_idx1, 0], X[max_dist_idx2, 0]], 
                      [X[max_dist_idx1, 1], X[max_dist_idx2, 1]], 
                      color='green', s=150, zorder=10)
            
        elif i == 2:  # Average - 绘制多条线表示平均距离
            # 仅显示部分线以避免图表过于混乱
            for a in range(0, 10, 3):
                for b in range(10, 20, 3):
                    ax.plot([X[a, 0], X[b, 0]], [X[a, 1], X[b, 1]], 'k:', lw=1, alpha=0.3)
            
            # 计算中心点
            center1 = np.mean(X[:10], axis=0)
            center2 = np.mean(X[10:], axis=0)
            ax.scatter(center1[0], center1[1], color='darkblue', s=200, marker='*', zorder=10)
            ax.scatter(center2[0], center2[1], color='darkred', s=200, marker='*', zorder=10)
            ax.plot([center1[0], center2[0]], [center1[1], center2[1]], 'k--', lw=2)
            
        elif i == 3:  # Ward - 显示簇内方差
            # 计算各自簇的中心
            center1 = np.mean(X[:10], axis=0)
            center2 = np.mean(X[10:], axis=0)
            
            # 显示中心点
            ax.scatter(center1[0], center1[1], color='darkblue', s=200, marker='*', zorder=10)
            ax.scatter(center2[0], center2[1], color='darkred', s=200, marker='*', zorder=10)
            
            # 显示到中心的距离
            for j in range(10):
                ax.plot([X[j, 0], center1[0]], [X[j, 1], center1[1]], 'b:', lw=1, alpha=0.3)
            for j in range(10, 20):
                ax.plot([X[j, 0], center2[0]], [X[j, 1], center2[1]], 'r:', lw=1, alpha=0.3)
                
            # 计算合并后的中心
            center_all = np.mean(X, axis=0)
            ax.scatter(center_all[0], center_all[1], color='purple', s=250, marker='*', zorder=10)
            
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True)
        
    plt.tight_layout()
    return fig

# 添加连接标准的图形说明
linkage_fig = create_linkage_comparison()
st.pyplot(linkage_fig)
st.caption("不同连接标准的直观对比 - 展示了各连接方式计算簇间距离的差异")

# 树状图
st.subheader("树状图 (Dendrogram)")
st.markdown("""
* 层次聚类的结果通常用树状图可视化
* 纵轴表示簇合并时的距离（或不相似度）
* 横轴表示数据点（或样本索引）
* 通过在某个距离阈值水平切割树状图，可以得到指定数量的簇
* 切割线穿过的垂直线数量即为簇的数量
* 树状图的高度差可以反映簇之间的分离程度
""")

# 创建自己的树状图示例
def create_dendrogram_example():
    set_chinese_font()  # 设置中文字体
    
    # 生成数据
    np.random.seed(42)
    X = np.vstack([
        np.random.normal(loc=[2, 2], scale=0.3, size=(5, 2)),  # 第一簇
        np.random.normal(loc=[7, 7], scale=0.3, size=(5, 2)),  # 第二簇
        np.random.normal(loc=[4.5, 8], scale=0.3, size=(5, 2))  # 第三簇
    ])
    
    # 计算链接矩阵
    from scipy.cluster.hierarchy import linkage, dendrogram
    linked = linkage(X, method='ward')
    
    # 绘制树状图
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(linked, ax=ax)
    
    # 添加切割线示意
    ax.axhline(y=3.5, color='r', linestyle='--')
    ax.text(16, 3.7, '切割线 (3个簇)', color='red', fontsize=12)
    
    # 添加标注
    ax.text(7, 6.5, '高度差大\n→簇间差异大', fontsize=12)
    ax.text(12, 2, '高度差小\n→簇间差异小', fontsize=12)
    
    ax.set_title('层次聚类树状图示例', fontsize=14)
    ax.set_xlabel('样本索引', fontsize=12)
    ax.set_ylabel('距离 (Ward)', fontsize=12)
    
    return fig

dendrogram_fig = create_dendrogram_example()
st.pyplot(dendrogram_fig)
st.caption("层次聚类树状图示例 - 展示了簇的合并过程和结构")

# 优缺点
st.header("优缺点对比")
adv_col1, adv_col2 = st.columns(2)

with adv_col1:
    st.subheader("优点")
    st.markdown("""
    * **无需预先指定K值**：可以根据树状图决定合适的簇数量
    * **可以揭示数据的层次结构**：树状图本身提供了丰富的结构信息
    * **可以发现非凸形状的簇**：使用某些连接标准（如Single Linkage）时
    * **确定性算法**：多次运行得到相同结果，不依赖初始化
    """)

with adv_col2:
    st.subheader("缺点")
    st.markdown("""
    * **计算复杂度较高**：通常为O(n³)或O(n²log n)，不适合非常大的数据集
    * **合并决策不可撤销**：一旦两个簇被合并，后续步骤无法撤销
    * **对距离度量和连接标准的选择敏感**：不同选择可能导致显著不同的结果
    * **存储需求大**：需要存储距离矩阵，对大型数据集来说可能是个问题
    """)

# 添加页脚
st.markdown("---")
st.markdown("© 2024 机器学习课程 | 交互式聚类算法课件") 