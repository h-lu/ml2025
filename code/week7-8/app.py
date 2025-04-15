import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import matplotlib as mpl

# 设置matplotlib支持中文显示
# 根据操作系统设置合适的中文字体
system = platform.system()
if system == 'Darwin':  # macOS
    plt.rcParams['font.family'] = ['Arial Unicode MS', 'PingFang SC', 'STHeiti', 'Heiti TC', 'sans-serif']
elif system == 'Windows':
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'sans-serif']
else:  # Linux或其他系统
    plt.rcParams['font.family'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'sans-serif']

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

from data_generator import generate_blob_data, generate_moon_data
from kmeans_demo import kmeans_demo
from dbscan_demo import dbscan_demo
from clustering_comparison import compare_clustering
from business_insights import business_insights_demo

st.set_page_config(
    page_title="无监督学习聚类算法演示",
    page_icon="📊",
    layout="wide"
)

st.title("无监督学习聚类算法演示")
st.caption("基于第七周和第八周讲座内容")

# 侧边栏导航
st.sidebar.title("导航")
page = st.sidebar.radio(
    "选择一个模块:",
    [
        "1️⃣ 简介与数据生成",
        "2️⃣ K-Means聚类",
        "3️⃣ DBSCAN聚类", 
        "4️⃣ 聚类算法比较",
        "5️⃣ 业务洞察解读"
    ]
)

# 简介页面
if page == "1️⃣ 简介与数据生成":
    st.header("无监督学习与聚类简介")
    
    st.subheader("无监督学习")
    st.write("""
    无监督学习的核心在于探索数据的内在结构，而不需要预先定义的标签。常见的无监督学习任务包括：
    - **聚类 (Clustering):** 将相似的数据点分到同一个组（簇），将不相似的数据点分到不同的组。
    - **降维 (Dimensionality Reduction):** 在保留数据主要信息的前提下，减少数据的特征数量。
    - **关联规则挖掘:** 发现数据项之间的有趣关联。
    """)
    
    st.subheader("聚类应用")
    st.write("""
    - **用户分群/市场细分:** 将具有相似特征或行为的用户划分到不同群体，以便进行精准营销。
    - **图像分割:** 将图像中像素根据颜色、纹理等特征聚类，以识别不同区域。
    - **异常检测:** 正常的数据点会聚集在一起，而异常点则会远离这些簇。
    - **文档分组:** 将内容相似的文档自动归类。
    """)
    
    st.subheader("数据生成工具")
    
    data_type = st.selectbox(
        "选择数据类型:",
        ["Blob数据 (适合K-Means)", "Moon数据 (适合DBSCAN)"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if data_type == "Blob数据 (适合K-Means)":
            n_samples = st.slider("样本数量", 100, 1000, 300)
            n_centers = st.slider("簇数量", 2, 8, 4)
            cluster_std = st.slider("簇标准差", 0.1, 2.0, 0.8)
            
            if st.button("生成Blob数据"):
                X, y = generate_blob_data(n_samples, n_centers, cluster_std)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)
                ax.set_title(f"生成的Blob数据 ({n_centers}个簇)")
                ax.set_xlabel("特征1")
                ax.set_ylabel("特征2")
                legend = ax.legend(*scatter.legend_elements(), title="簇")
                st.pyplot(fig)
        else:
            n_samples = st.slider("样本数量", 100, 1000, 300)
            noise = st.slider("噪声程度", 0.01, 0.2, 0.05)
            
            if st.button("生成Moon数据"):
                X, y = generate_moon_data(n_samples, noise)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)
                ax.set_title("生成的Moon数据")
                ax.set_xlabel("特征1")
                ax.set_ylabel("特征2")
                legend = ax.legend(*scatter.legend_elements(), title="簇")
                st.pyplot(fig)
    
    with col2:
        st.write("""
        ### 数据特点:
        
        **Blob数据**:
        - 呈球状分布
        - 各簇大小较为均匀
        - 簇边界相对明显
        - 非常适合K-Means算法
        
        **Moon数据**:
        - 呈新月形状分布
        - 非凸形状
        - 簇边界复杂
        - 更适合DBSCAN算法
        
        > 注意: 聚类算法的选择取决于数据的分布特征和业务需求!
        """)

# K-Means演示页面
elif page == "2️⃣ K-Means聚类":
    kmeans_demo()

# DBSCAN演示页面
elif page == "3️⃣ DBSCAN聚类":
    dbscan_demo()

# 聚类算法比较页面
elif page == "4️⃣ 聚类算法比较":
    compare_clustering()

# 业务洞察解读页面
elif page == "5️⃣ 业务洞察解读":
    business_insights_demo()

# 底部信息
st.sidebar.markdown("---")
st.sidebar.caption("基于2024春季学期机器学习课程") 