import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from data_generator import generate_moon_data, generate_varied_blobs

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

def dbscan_demo():
    """DBSCAN聚类算法演示页面"""
    st.header("DBSCAN聚类算法")
    
    # DBSCAN原理说明
    with st.expander("DBSCAN算法原理", expanded=True):
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.write("""
            ### DBSCAN工作原理
            
            DBSCAN(密度聚类)是一种基于**密度**的聚类算法，由Ester等人于1996年提出。
            它不需要预先指定簇的数量，并且能够识别任意形状的簇和噪声点。
            
            **核心概念:**
            
            - **ε (epsilon/eps):** 邻域半径，定义样本点的"邻域"范围
            - **MinPts (min_samples):** 成为核心点所需的最小样本数
            - **核心点 (Core Point):** 其ε-邻域内至少有MinPts个样本的点
            - **边界点 (Border Point):** 落在某个核心点邻域内但自身不是核心点的样本点
            - **噪声点 (Noise Point):** 既不是核心点也不是边界点的样本点
            
            **算法步骤:**
            1. 随机选择一个未访问过的样本点P
            2. 如果P是核心点，创建一个新簇，并将其邻域内所有点加入该簇
            3. 递归地将新加入簇的每一个核心点的邻域加入该簇
            4. 当没有新点可加入当前簇时，选择另一个未访问点重复步骤1-3
            5. 直到所有点被访问，算法结束
            """)
        
        with col2:
            st.image("https://upload.wikimedia.org/wikipedia/commons/0/05/DBSCAN-density-data.svg", 
                    caption="DBSCAN核心概念示意图")
    
    # 数据生成部分
    st.subheader("生成数据")
    
    data_type = st.radio(
        "选择数据类型:",
        ["Moon数据 (非凸形状)", "混合密度数据 (不同密度)"]
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if data_type == "Moon数据 (非凸形状)":
            n_samples = st.slider("样本数量", 100, 1000, 300, key="dbscan_n_samples")
            noise = st.slider("噪声程度", 0.01, 0.2, 0.05, key="dbscan_noise")
            
            if "dbscan_data" not in st.session_state or st.session_state.dbscan_data_type != "moon":
                st.session_state.dbscan_data = generate_moon_data(n_samples, noise)
                st.session_state.dbscan_data_type = "moon"
            
            if st.button("重新生成数据"):
                st.session_state.dbscan_data = generate_moon_data(n_samples, noise)
                st.session_state.dbscan_data_type = "moon"
        else:
            n_samples = st.slider("样本数量", 200, 1000, 500, key="dbscan_varied_n_samples")
            
            if "dbscan_data" not in st.session_state or st.session_state.dbscan_data_type != "varied":
                st.session_state.dbscan_data = generate_varied_blobs(n_samples)
                st.session_state.dbscan_data_type = "varied"
            
            if st.button("重新生成数据"):
                st.session_state.dbscan_data = generate_varied_blobs(n_samples)
                st.session_state.dbscan_data_type = "varied"
    
    # 展示生成的数据
    X, y_true = st.session_state.dbscan_data
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=50, alpha=0.7)
        ax.set_title("生成的数据")
        ax.set_xlabel("特征1")
        ax.set_ylabel("特征2")
        legend = ax.legend(*scatter.legend_elements(), title="真实簇")
        ax.add_artist(legend)
        st.pyplot(fig)
    
    # 参数选择部分
    st.subheader("DBSCAN参数选择")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("""
        ### 参数选择指南
        
        DBSCAN算法有两个关键参数:
        
        1. **eps (ε):** 定义邻域半径
           - 太小: 会产生大量噪声点，或将一个簇分成多个
           - 太大: 可能会将不同簇合并
        
        2. **min_samples (MinPts):** 成为核心点所需的最小样本数
           - 通常建议设置为 ≥ 数据维度+1
           - 增大此值可减少噪声敏感性
        
        K-距离图可以帮助选择合适的eps值:
        - 选择min_samples
        - 计算每个点到其第k个最近邻的距离
        - 找到图中"拐点"对应的距离作为eps
        """)
    
    # K-距离图
    with col2:
        X = st.session_state.dbscan_data[0]
        
        min_pts = st.slider("选择min_samples值", 2, 20, 5, key="dbscan_min_pts")
        
        # 计算K-距离
        nbrs = NearestNeighbors(n_neighbors=min_pts).fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # 获取第k-1个邻居的距离
        k_distances = np.sort(distances[:, min_pts-1], axis=0)
        
        # 绘制K-距离图
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(k_distances)
        ax.set_title(f'{min_pts}-距离图')
        ax.set_xlabel("点的排序索引")
        ax.set_ylabel(f'第{min_pts}个最近邻点的距离')
        ax.grid(True)
        
        # 尝试自动找到"拐点"
        from scipy import interpolate
        from scipy.signal import argrelextrema
        
        # 使用样条插值平滑曲线
        x = np.arange(len(k_distances))
        x_smooth = np.linspace(0, len(k_distances)-1, 1000)
        try:
            # 使用更适合K-距离图的插值
            f = interpolate.interp1d(x, k_distances, kind='cubic')
            k_distances_smooth = f(x_smooth)
            
            # 尝试找到曲率变化较大的点
            # 计算导数和二阶导数
            dx1 = np.gradient(k_distances_smooth)
            dx2 = np.gradient(dx1)
            
            # 曲率 - 用于找到拐点
            curvature = np.abs(dx2) / (1 + dx1**2)**1.5
            
            # 找到前10%的曲率最大点
            n_points = int(0.1 * len(curvature))
            idx = np.argpartition(curvature, -n_points)[-n_points:]
            idx = idx[np.argsort(-curvature[idx])]
            
            # 选择在前50%的第一个拐点
            for i in idx:
                if i > len(x_smooth) * 0.1 and i < len(x_smooth) * 0.5:
                    elbow_idx = i
                    break
            else:
                # 如果找不到符合条件的点，选择最大曲率点
                elbow_idx = np.argmax(curvature[int(len(curvature)*0.1):int(len(curvature)*0.5)]) + int(len(curvature)*0.1)
            
            elbow_kdist = k_distances_smooth[elbow_idx]
            suggested_eps = round(elbow_kdist, 2)
            
            # 在图中标记建议的eps值
            ax.axhline(y=suggested_eps, color='r', linestyle='--', alpha=0.7)
            ax.text(len(k_distances) * 0.7, suggested_eps + 0.02, 
                   f'建议的eps值: {suggested_eps}', color='r')
        except:
            suggested_eps = 0.3  # 失败时的默认值
        
        st.pyplot(fig)
        
        st.markdown(f"**建议的eps值: {suggested_eps}**")
    
    # DBSCAN实际应用部分
    st.subheader("DBSCAN聚类演示")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        eps = st.slider("选择eps值", 0.01, 1.0, suggested_eps, 0.01, key="dbscan_eps")
        min_samples = st.slider("选择min_samples值", 2, 20, min_pts, key="dbscan_min_samples")
        
        st.write("""
        **参数说明:**
        - **eps:** 邻域半径，定义点之间的距离阈值
        - **min_samples:** 成为核心点所需的最小样本数
        
        **注意:**
        - 噪声点将被标记为-1
        - 有效簇从0开始编号
        """)
    
    # 运行DBSCAN并可视化
    with col2:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(X)
        labels = dbscan.labels_
        
        # 计算簇数量和噪声点数量
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        st.write(f"发现的簇数量: {n_clusters}")
        st.write(f"噪声点数量: {n_noise} ({n_noise/len(labels):.1%})")
        
        # 如果数据集有真实标签，计算调整兰德指数
        if 'y_true' in locals():
            from sklearn.metrics import adjusted_rand_score
            ari = adjusted_rand_score(y_true, labels)
            st.write(f"与真实簇的调整兰德指数: {ari:.4f}")
        
        # 计算当前参数的轮廓系数 (如果簇数量>1且不是所有点都是噪声)
        valid_clusters = len(set(labels) - {-1})
        if valid_clusters > 1 and n_noise < len(labels):
            # 过滤掉噪声点计算轮廓系数
            mask = labels != -1
            if np.sum(mask) > 1:  # 确保有足够的非噪声点
                try:
                    silhouette = silhouette_score(X[mask], labels[mask])
                    st.write(f"轮廓系数 (不含噪声点): {silhouette:.4f}")
                except:
                    st.write("无法计算轮廓系数 (可能每个簇只有一个样本)")
        
        # 可视化聚类结果
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 为噪声点创建特殊颜色映射
        unique_labels = set(labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels) - (1 if -1 in labels else 0)))
        color_dict = {}
        
        i = 0
        for label in unique_labels:
            if label == -1:
                color_dict[label] = (0.7, 0.7, 0.7, 1.0)  # 灰色表示噪声
            else:
                color_dict[label] = colors[i]
                i += 1
        
        for label in unique_labels:
            mask = labels == label
            if label == -1:
                ax.scatter(X[mask, 0], X[mask, 1], s=50, color=color_dict[label], 
                          alpha=0.5, label='噪声点')
            else:
                ax.scatter(X[mask, 0], X[mask, 1], s=50, color=color_dict[label], 
                          alpha=0.7, label=f'簇 {label}')
        
        ax.set_title(f"DBSCAN聚类结果 (eps={eps}, min_samples={min_samples})")
        ax.set_xlabel("特征1")
        ax.set_ylabel("特征2")
        ax.legend()
        
        st.pyplot(fig)
    
    # DBSCAN优缺点部分
    with st.expander("DBSCAN的优缺点"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            ### 优点:
            - **不需要指定簇数量K**
            - **能发现任意形状的簇**
            - **能识别噪声点/异常值**
            - **对簇的形状和大小不敏感**
            - **只需要两个参数(eps, min_samples)**
            """)
        
        with col2:
            st.write("""
            ### 缺点:
            - **对参数eps和min_samples非常敏感**
            - **对于密度差异很大的簇效果不佳**
            - **高维数据的性能可能下降**
            - **计算复杂度相对较高**
            - **对大规模数据处理能力有限**
            """) 