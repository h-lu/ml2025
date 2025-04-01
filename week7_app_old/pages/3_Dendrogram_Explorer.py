import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
import time
from utils.font_manager import set_chinese_font

st.set_page_config(
    page_title="树状图探索器",
    page_icon="🌳",
    layout="wide"
)

st.title("层次聚类树状图探索器")

st.markdown("""
本页面允许您通过调整参数来生成不同的数据集，并观察在不同连接标准下的层次聚类树状图。
您还可以通过设置切割线来模拟选择簇的数量。
""")

# 创建会话状态变量来保存数据和参数
if 'X' not in st.session_state:
    st.session_state.X = None
if 'cluster_std_all' not in st.session_state:
    st.session_state.cluster_std_all = None
if 'centers' not in st.session_state:
    st.session_state.centers = None
if 'random_state' not in st.session_state:
    st.session_state.random_state = 42
if 'cut_height' not in st.session_state:
    st.session_state.cut_height = 10.0  # 确保这是浮点数

# 侧边栏设置
st.sidebar.header("数据生成参数")

# 数据生成参数
n_samples = st.sidebar.slider("样本数量", min_value=20, max_value=200, value=50, step=10)
n_centers = st.sidebar.slider("簇中心数量", min_value=2, max_value=8, value=3)

# 允许为每个簇设置不同的标准差
use_different_std = st.sidebar.checkbox("为每个簇设置不同的标准差", value=False)

if use_different_std:
    st.sidebar.subheader("每个簇的标准差")
    cluster_stds = []
    for i in range(n_centers):
        std = st.sidebar.slider(f"簇 {i+1} 的标准差", 
                             min_value=0.1, max_value=3.0, value=1.0, step=0.1)
        cluster_stds.append(std)
    cluster_std = cluster_stds
else:
    cluster_std = st.sidebar.slider("簇标准差", min_value=0.1, max_value=3.0, value=1.0, step=0.1)

# 随机种子设置
random_state = st.sidebar.slider("随机种子", min_value=0, max_value=100, value=st.session_state.random_state)

# 连接标准选择
st.sidebar.header("层次聚类参数")
linkage_method = st.sidebar.selectbox(
    "连接标准", 
    options=["ward", "average", "complete", "single"],
    format_func=lambda x: {
        "ward": "Ward (最小方差增量)",
        "average": "Average (平均距离)",
        "complete": "Complete (最大距离)",
        "single": "Single (最小距离)"
    }.get(x, x)
)

# 生成数据按钮
if st.sidebar.button("生成新数据"):
    st.session_state.random_state = random_state
    X, y = make_blobs(n_samples=n_samples, 
                    centers=n_centers, 
                    cluster_std=cluster_std,
                    random_state=random_state)
    st.session_state.X = X
    st.session_state.cluster_std_all = cluster_std
    st.session_state.centers = n_centers
    # 保存连接方法
    st.session_state.linkage_method = linkage_method

# 切割高度选择
cut_height = st.slider(
    "设置树状图切割高度（用于确定簇的数量）", 
    min_value=0.0, 
    max_value=50.0, 
    value=float(st.session_state.cut_height),  # 确保使用浮点数
    step=0.5
)
st.session_state.cut_height = cut_height

# 主要内容
main_col1, main_col2 = st.columns([3, 2])

# 如果数据已生成，显示数据集散点图和树状图
if st.session_state.X is not None:
    X = st.session_state.X
    
    with main_col1:
        st.subheader("层次聚类树状图")
        
        # 计算连接矩阵和绘制树状图
        with st.spinner("计算连接矩阵并绘制树状图..."):
            # 计时开始
            start_time = time.time()
            
            # 设置中文字体
            set_chinese_font()
            
            # 计算连接矩阵
            linked = linkage(X, method=linkage_method)
            
            # 创建树状图
            fig, ax = plt.subplots(figsize=(10, 6))
            dendrogram(linked,
                      orientation='top',
                      distance_sort='descending',
                      show_leaf_counts=True,
                      ax=ax)
            
            # 添加切割线并计算簇的数量
            plt.axhline(y=cut_height, color='r', linestyle='--')
            
            # 计算切割线对应的簇数量
            k = len(set(list(map(lambda x: x[0], (linked[:, :2][linked[:, 2] >= cut_height])))))
            
            # 设置标题和标签
            plt.title(f'层次聚类树状图 ({linkage_method.capitalize()} 连接)')
            plt.xlabel('样本索引')
            plt.ylabel(f'距离 ({linkage_method.capitalize()})')
            
            # 计时结束
            end_time = time.time()
            
            st.pyplot(fig)
            
            st.info(f"切割高度 {cut_height:.2f} 对应的簇数量为: **{k}**")
            st.text(f"计算耗时: {end_time - start_time:.4f} 秒")
            
            # 保存簇的数量供后续页面使用
            st.session_state.k_clusters = k
    
    with main_col2:
        st.subheader("数据散点图")
        
        # 绘制数据散点图
        set_chinese_font()  # 设置中文字体
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X[:, 0], X[:, 1], s=50, alpha=0.8)
        plt.title(f"生成的数据集 ({n_samples} 个样本, {st.session_state.centers} 个中心)")
        plt.xlabel("特征 1")
        plt.ylabel("特征 2")
        plt.grid(True)
        st.pyplot(fig)
        
        # 显示数据生成参数
        st.markdown("#### 数据生成参数")
        st.markdown(f"""
        * **样本数量:** {n_samples}
        * **簇中心数量:** {st.session_state.centers}
        * **簇标准差:** {st.session_state.cluster_std_all}
        * **随机种子:** {st.session_state.random_state}
        """)
        
        # 提示用户下一步操作
        st.success("您可以调整侧边栏中的参数重新生成数据，或调整切割高度来获得不同数量的簇。")
        st.info("在下一页'层次聚类实践'中，您可以使用选定的簇数量应用AgglomerativeClustering算法。")
else:
    # 如果数据未生成，显示提示
    st.info("请在侧边栏中设置参数并点击'生成新数据'按钮。")

# 添加树状图解读指南
with st.expander("树状图解读指南"):
    st.markdown("""
    ### 如何解读树状图 (Dendrogram)

    1. **垂直线**：代表一个簇。高度越高，簇包含的点越多。
    2. **水平线**：表示簇的合并。水平线的高度表示合并时的距离（不相似度）。
    3. **红色虚线**：切割线，确定簇的数量。线与垂直线相交的次数即为簇的数量。
    4. **距离高度**：
       - 短距离：表示相似的簇被合并
       - 长距离：表示不相似的簇被合并
       - 高度差大：表示簇之间的分离明显
       - 高度差小：表示簇之间的分离不明显
    
    ### 不同连接标准的特点

    1. **Ward连接**：最小化簇内方差的增量，倾向于产生大小相似的簇。
    2. **Average连接**：使用簇间点对的平均距离，是一种相对平衡的方法。
    3. **Complete连接**：使用簇间点对的最大距离，倾向于产生紧凑的球状簇。
    4. **Single连接**：使用簇间点对的最小距离，可以找到非凸形状的簇，但易受噪声影响，产生"链式效应"。
    """)

# 添加页脚
st.markdown("---")
st.markdown("© 2024 机器学习课程 | 交互式聚类算法课件") 