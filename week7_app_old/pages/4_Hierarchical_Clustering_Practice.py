import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import time
from utils.font_manager import set_chinese_font

st.set_page_config(
    page_title="层次聚类实践",
    page_icon="🧪",
    layout="wide"
)

st.title("层次聚类实践")

st.markdown("""
本页面将使用上一页中生成的数据集和设置的参数，应用AgglomerativeClustering算法进行聚类，并可视化结果。
您可以选择簇的数量和连接标准，观察不同设置下的聚类效果。
""")

# 检查是否已有数据
if 'X' not in st.session_state or st.session_state.X is None:
    st.warning("您尚未生成数据。请先前往'树状图探索器'页面生成数据。")
    st.stop()

# 获取数据和参数
X = st.session_state.X

# 侧边栏设置
st.sidebar.header("层次聚类参数")

# 获取上一页设置的簇数量，如果没有则默认为3
if 'k_clusters' in st.session_state:
    default_k = st.session_state.k_clusters
else:
    default_k = 3

# 簇数量选择
n_clusters = st.sidebar.slider("簇数量", min_value=2, max_value=10, value=default_k)

# 连接标准选择
linkage_options = ["ward", "average", "complete", "single"]
linkage_labels = {
    "ward": "Ward (最小方差增量)",
    "average": "Average (平均距离)",
    "complete": "Complete (最大距离)",
    "single": "Single (最小距离)"
}

# 如果上一页设置了连接标准，则使用该标准作为默认值
if 'linkage_method' in st.session_state:
    default_linkage = st.session_state.linkage_method
else:
    default_linkage = "ward"

# 如果默认连接标准不在选项中，则使用第一个选项
if default_linkage not in linkage_options:
    default_linkage = linkage_options[0]

linkage_method = st.sidebar.selectbox(
    "连接标准", 
    options=linkage_options,
    index=linkage_options.index(default_linkage),
    format_func=lambda x: linkage_labels.get(x, x)
)

# 是否显示代码
show_code = st.sidebar.checkbox("显示Python代码", value=False)

# 运行聚类按钮
if st.sidebar.button("运行层次聚类"):
    st.session_state.run_clustering = True
    # 保存选择的连接标准到会话状态，以便其他页面使用
    st.session_state.linkage_method = linkage_method

# 主要内容部分
col1, col2 = st.columns([3, 2])

# 运行聚类和显示结果
if 'run_clustering' in st.session_state and st.session_state.run_clustering:
    with col1:
        st.subheader("聚类结果可视化")
        
        with st.spinner("正在执行层次聚类..."):
            # 计时开始
            start_time = time.time()
            
            # 设置中文字体
            set_chinese_font()
            
            # 创建并训练模型
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage_method,
                compute_distances=True  # 计算距离矩阵（用于绘制树状图）
            )
            
            cluster_labels = model.fit_predict(X)
            
            # 计时结束
            end_time = time.time()
            
            # 绘制聚类结果
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 绘制散点图，颜色根据簇标签
            scatter = ax.scatter(X[:, 0], X[:, 1], c=cluster_labels, 
                      cmap='viridis', s=50, alpha=0.8)
            
            # 添加图例
            legend = ax.legend(*scatter.legend_elements(),
                            title="簇")
            ax.add_artist(legend)
            
            # 设置标题和标签
            ax.set_title(f'层次聚类结果 (k={n_clusters}, {linkage_method}连接)')
            ax.set_xlabel('特征 1')
            ax.set_ylabel('特征 2')
            ax.grid(True)
            
            # 显示图形
            st.pyplot(fig)
            
            # 显示计算时间
            st.text(f"聚类计算耗时: {end_time - start_time:.4f} 秒")
            
            # 显示簇的统计信息
            st.subheader("簇的统计信息")
            counts = np.bincount(cluster_labels)
            stats_data = {
                "簇标签": list(range(len(counts))),
                "样本数量": counts,
                "百分比": [f"{count/len(cluster_labels)*100:.2f}%" for count in counts]
            }
            
            # 创建一个表格显示簇的统计信息
            st.table(stats_data)
    
    with col2:
        st.subheader("实验参数")
        st.markdown(f"""
        #### 数据信息
        * **样本数量:** {X.shape[0]}
        * **特征维度:** {X.shape[1]}
        
        #### 聚类参数
        * **簇数量 (k):** {n_clusters}
        * **连接标准:** {linkage_labels[linkage_method]}
        """)
        
        # 显示Python代码
        if show_code:
            st.subheader("Python代码示例")
            
            code = f'''
# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# 创建层次聚类模型
model = AgglomerativeClustering(
    n_clusters={n_clusters},
    linkage="{linkage_method}"
)

# 训练模型并预测簇标签
cluster_labels = model.fit_predict(X)

# 可视化结果
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, 
                   cmap='viridis', s=50, alpha=0.8)
plt.title(f'层次聚类结果 (k={n_clusters}, {linkage_method}连接)')
plt.xlabel('特征 1')
plt.ylabel('特征 2')
plt.grid(True)
plt.legend(*scatter.legend_elements(), title="簇")
plt.show()
'''
            st.code(code, language='python')
        
        # 添加解释和注意事项
        st.info("您可以通过调整侧边栏中的参数来改变聚类结果，例如尝试不同的簇数量和连接标准。")
        
        with st.expander("不同连接标准的聚类特点"):
            st.markdown("""
            ### 不同连接标准对聚类结果的影响
            
            * **Ward连接**：倾向于产生大小相似的紧凑球状簇。适合大多数常见场景。
            * **Average连接**：较为平衡，对异常值不太敏感。
            * **Complete连接**：倾向于产生紧凑的球状簇，可能导致簇大小不均衡。
            * **Single连接**：能够识别非凸形状的簇，但容易受噪声影响，产生链状效应。
            
            实际应用中，Ward连接通常是默认选择，但对于特定数据，其他连接标准可能效果更好。建议尝试多种连接标准并比较结果。
            """)
else:
    # 如果尚未运行聚类，显示提示
    st.info("请在侧边栏中设置参数，然后点击'运行层次聚类'按钮。")

# 添加页脚
st.markdown("---")
st.markdown("© 2024 机器学习课程 | 交互式聚类算法课件") 