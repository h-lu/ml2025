import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from data_generator import generate_blob_data

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

def kmeans_demo():
    """K-Means聚类算法演示页面"""
    st.header("K-Means聚类算法")
    
    # K-Means原理说明
    with st.expander("K-Means算法原理", expanded=True):
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.write("""
            ### K-Means工作原理
            
            K-Means是最简单、最常用的聚类算法之一，由MacQueen于1967年提出。它的目标是将数据集划分为K个预先指定的簇。

            **算法步骤:**
            1. **指定K值:** 人为指定想要划分的簇数量K
            2. **随机初始化质心:** 随机选择K个数据点作为初始的簇质心
            3. **分配样本:** 将每个样本分配给距离最近的质心所代表的簇
            4. **更新质心:** 重新计算每个簇所有成员样本的平均值作为新质心
            5. **迭代:** 重复步骤3和4，直到满足停止条件
            
            **停止条件:**
            - 质心不再发生明显变化
            - 样本点所属的簇不再变化
            - 达到最大迭代次数
            """)
        
        with col2:
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/K-means_convergence.gif/300px-K-means_convergence.gif", 
                    caption="K-Means收敛过程")
    
    # 数据生成部分
    st.subheader("生成数据")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        n_samples = st.slider("样本数量", 100, 1000, 300, key="kmeans_n_samples")
        n_centers = st.slider("真实簇数量", 2, 8, 4, key="kmeans_n_centers")
        cluster_std = st.slider("簇标准差", 0.1, 2.0, 0.8, key="kmeans_std")
        
        if "kmeans_data" not in st.session_state:
            st.session_state.kmeans_data = generate_blob_data(
                n_samples, n_centers, cluster_std)
        
        if st.button("重新生成数据"):
            st.session_state.kmeans_data = generate_blob_data(
                n_samples, n_centers, cluster_std)
    
    # 展示生成的数据
    X, y_true = st.session_state.kmeans_data
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=50, alpha=0.7)
        ax.set_title(f"生成的数据 (真实簇数量: {n_centers})")
        ax.set_xlabel("特征1")
        ax.set_ylabel("特征2")
        legend = ax.legend(*scatter.legend_elements(), title="真实簇")
        ax.add_artist(legend)
        st.pyplot(fig)
    
    # K-Means的K值选择部分
    st.subheader("K值选择")
    k_method_tab1, k_method_tab2 = st.tabs(["肘部法则", "轮廓系数"])
    
    # 提前计算不同K值的结果
    k_range = range(1, 11)
    inertias = []
    silhouette_scores = []
    
    X = st.session_state.kmeans_data[0]
    
    with st.spinner("计算不同K值的评估指标..."):
        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            
            # 轮廓系数至少需要2个簇
            if k > 1:
                score = silhouette_score(X, kmeans.labels_)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(0)  # K=1时无法计算轮廓系数
    
    # 肘部法则标签页
    with k_method_tab1:
        st.write("""
        **肘部法则 (Elbow Method)** 是选择K值的一种常用方法:
        
        1. 尝试不同的K值
        2. 计算每个K值下的簇内平方和(WCSS/Inertia)
        3. 绘制K值与Inertia的关系图
        4. 寻找图中曲线"肘部"处的K值
        
        > **原理:** 当K小于真实簇数时，增加K会显著提高簇内紧密度；当K超过真实簇数时，再增加K对Inertia的改善效果会变得不明显。
        """)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_range, inertias, 'o-')
        ax.set_title("肘部法则")
        ax.set_xlabel("簇数量 (K)")
        ax.set_ylabel("Inertia")
        ax.grid(True)
        
        # 找到可能的肘部点
        from scipy.signal import argrelextrema
        from scipy import interpolate
        
        # 使用样条插值平滑曲线
        x_smooth = np.linspace(1, 10, 100)
        f = interpolate.interp1d(k_range, inertias, kind='cubic')
        inertias_smooth = f(x_smooth)
        
        # 计算曲率
        dx1 = np.gradient(x_smooth)
        dy1 = np.gradient(inertias_smooth)
        dx2 = np.gradient(dx1)
        dy2 = np.gradient(dy1)
        curvature = np.abs(dx1 * dy2 - dx2 * dy1) / (dx1**2 + dy1**2)**1.5
        
        # 找到曲率最大的点作为可能的肘部
        elbow_idx = np.argmax(curvature)
        elbow_k = x_smooth[elbow_idx]
        
        # 在图中标记肘部点
        if 1 < elbow_k < 10:
            ax.axvline(x=round(elbow_k), color='r', linestyle='--', alpha=0.7)
            ax.text(round(elbow_k) + 0.1, inertias[round(elbow_k)-1], 
                   f'可能的肘部: K={round(elbow_k)}', color='r')
        
        st.pyplot(fig)
        
        st.write(f"根据肘部法则，K = {round(elbow_k)} 可能是一个合适的簇数量。")
    
    # 轮廓系数标签页
    with k_method_tab2:
        st.write("""
        **轮廓系数 (Silhouette Score)** 衡量样本与自身簇的紧密度与其他簇的分离度:
        
        1. 计算样本i与同簇其他样本的平均距离a(i)
        2. 计算样本i与最近其他簇所有样本的平均距离b(i)
        3. 样本i的轮廓系数: s(i) = (b(i) - a(i)) / max(a(i), b(i))
        4. 整体轮廓系数是所有样本轮廓系数的平均值
        
        > **取值范围:** -1到1。接近1表示聚类效果好；接近0表示样本在两个簇的边界上；接近-1表示可能被分到了错误的簇。
        """)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        # 从K=2开始绘制轮廓系数
        ax.plot(list(k_range)[1:], silhouette_scores[1:], 'o-')
        ax.set_title("轮廓系数法")
        ax.set_xlabel("簇数量 (K)")
        ax.set_ylabel("轮廓系数")
        ax.grid(True)
        
        # 找到最佳K值
        best_k_idx = np.argmax(silhouette_scores[1:]) + 1  # 加1是因为我们从K=2开始计算
        best_k = k_range[best_k_idx]
        
        # 在图中标记最佳K值
        ax.axvline(x=best_k, color='r', linestyle='--', alpha=0.7)
        ax.text(best_k + 0.1, silhouette_scores[best_k_idx], 
                f'最佳K值: K={best_k}', color='r')
        
        st.pyplot(fig)
        
        st.write(f"根据轮廓系数，K = {best_k} 可能是一个最佳的簇数量。")
    
    # K-Means实际应用部分
    st.subheader("K-Means聚类演示")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_k = st.slider("选择K值", 2, 10, best_k if 'best_k' in locals() else 4)
        random_init = st.checkbox("随机初始化质心", False)
        init_method = 'random' if random_init else 'k-means++'
        n_init = st.slider("初始化次数", 1, 20, 10)
        
        st.write("""
        **参数说明:**
        - **K值:** 指定簇数量
        - **初始化方法:** 
          - k-means++更智能地选择初始质心
          - random完全随机选择初始质心
        - **初始化次数:** 用不同初始值运行算法的次数，选择最优结果
        """)
    
    # 运行K-Means并可视化
    with col2:
        kmeans = KMeans(
            n_clusters=selected_k, 
            init=init_method, 
            n_init=n_init, 
            random_state=42
        )
        
        kmeans.fit(X)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        
        # 如果数据集有真实标签，计算调整兰德指数
        if 'y_true' in locals():
            from sklearn.metrics import adjusted_rand_score
            ari = adjusted_rand_score(y_true, labels)
            st.write(f"与真实簇的调整兰德指数: {ari:.4f}")
        
        # 计算当前K值的轮廓系数
        if selected_k > 1:
            silhouette = silhouette_score(X, labels)
            st.write(f"轮廓系数: {silhouette:.4f}")
        
        # 可视化聚类结果
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
        ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='质心')
        ax.set_title(f"K-Means聚类结果 (K={selected_k})")
        ax.set_xlabel("特征1")
        ax.set_ylabel("特征2")
        legend1 = ax.legend(*scatter.legend_elements(), title="簇")
        ax.add_artist(legend1)
        ax.legend(loc='upper left')
        
        st.pyplot(fig)
    
    # K-Means优缺点部分
    with st.expander("K-Means的优缺点"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            ### 优点:
            - **简单高效:** 算法原理简单，计算速度快
            - **易于理解和实现**
            - **适用于大数据集:** 计算复杂度相对较低
            - **效果尚可:** 在簇呈凸形且大小相似时效果好
            """)
        
        with col2:
            st.write("""
            ### 缺点:
            - **需要预先指定K值**
            - **对初始质心敏感**
            - **对异常值敏感**
            - **倾向于发现球状簇:** 难以处理复杂形状的簇
            - **基于距离:** 必须进行特征缩放
            - **无法处理噪声点:** 所有点都会被分配到某个簇
            """) 