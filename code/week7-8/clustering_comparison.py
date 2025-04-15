import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

def make_complex_dataset(n_samples=1000, random_state=42):
    """生成复杂的混合数据集，用于比较聚类算法效果"""
    
    rng = np.random.RandomState(random_state)
    
    # 生成2个高斯簇
    X1, y1 = make_blobs(
        n_samples=int(0.3 * n_samples),
        centers=[[0, 0], [1, 5]],
        cluster_std=[0.5, 0.5],
        random_state=random_state
    )
    
    # 生成2个圆圈
    X2, y2 = make_circles(
        n_samples=int(0.4 * n_samples),
        noise=0.05,
        factor=0.5,
        random_state=random_state
    )
    # 移动圆圈位置并调整标签
    X2[:, 0] += 5
    X2[:, 1] += 2
    y2 += 2
    
    # 生成一个半月形
    X3, y3 = make_moons(
        n_samples=int(0.3 * n_samples),
        noise=0.05,
        random_state=random_state
    )
    # 移动半月形位置并调整标签
    X3[:, 0] += 3
    X3[:, 1] -= 3
    y3 += 4
    
    # 添加一些噪声点
    n_noise = int(0.05 * n_samples)
    X_noise = rng.uniform(-2, 7, (n_noise, 2))
    y_noise = np.full(n_noise, -1)  # 噪声标记为-1
    
    # 合并数据
    X = np.vstack([X1, X2, X3, X_noise])
    y = np.hstack([y1, y2, y3, y_noise])
    
    # 特征缩放
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def compare_clustering():
    """聚类算法比较演示页面"""
    st.header("K-Means vs. DBSCAN 算法比较")
    
    st.write("""
    本页面将直观展示K-Means和DBSCAN聚类算法各自的优势和局限性。通过在不同形状的数据集上应用这两种算法，我们可以清晰地看出它们的适用场景差异。
    """)
    
    # 数据集选择部分
    st.subheader("选择数据集")
    
    dataset_type = st.selectbox(
        "数据集类型:",
        ["球状数据 (适合K-Means)", "非凸形数据 (适合DBSCAN)", "混合复杂数据", "有噪声的数据"]
    )
    
    # 根据选择生成数据
    if dataset_type == "球状数据 (适合K-Means)":
        n_samples = st.slider("样本数量", 100, 1000, 300, key="comp_blob_samples")
        n_centers = st.slider("簇数量", 2, 8, 4, key="comp_blob_centers")
        cluster_std = st.slider("簇标准差", 0.1, 2.0, 0.8, key="comp_blob_std")
        
        X, y_true = make_blobs(
            n_samples=n_samples, 
            centers=n_centers, 
            cluster_std=cluster_std,
            random_state=42
        )
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        title = f"球状数据 ({n_centers}个簇)"
        
    elif dataset_type == "非凸形数据 (适合DBSCAN)":
        data_shape = st.radio("形状:", ["新月形", "圆环形"])
        n_samples = st.slider("样本数量", 100, 1000, 300, key="comp_nonconvex_samples")
        noise = st.slider("噪声程度", 0.01, 0.2, 0.05, key="comp_nonconvex_noise")
        
        if data_shape == "新月形":
            X, y_true = make_moons(
                n_samples=n_samples,
                noise=noise,
                random_state=42
            )
            title = "新月形数据"
        else:
            X, y_true = make_circles(
                n_samples=n_samples,
                noise=noise,
                factor=0.5,
                random_state=42
            )
            title = "圆环形数据"
            
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    elif dataset_type == "混合复杂数据":
        n_samples = st.slider("样本数量", 500, 2000, 1000, key="comp_complex_samples")
        X, y_true = make_complex_dataset(n_samples=n_samples)
        title = "混合复杂数据"
        
    else:  # 有噪声的数据
        n_samples = st.slider("样本数量", 100, 1000, 300, key="comp_noise_samples")
        n_centers = st.slider("簇数量", 2, 5, 3, key="comp_noise_centers")
        noise_ratio = st.slider("噪声比例", 0.05, 0.5, 0.2, key="comp_noise_ratio")
        
        # 生成正常簇
        X_normal, y_normal = make_blobs(
            n_samples=int(n_samples * (1-noise_ratio)), 
            centers=n_centers,
            cluster_std=0.7,
            random_state=42
        )
        
        # 生成噪声点
        rng = np.random.RandomState(42)
        n_noise = int(n_samples * noise_ratio)
        X_noise = rng.uniform(-5, 5, (n_noise, 2))
        y_noise = np.full(n_noise, -1)  # 噪声标记为-1
        
        # 合并数据
        X = np.vstack([X_normal, X_noise])
        y_true = np.hstack([y_normal, y_noise])
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        title = f"有噪声的数据 (噪声比例: {noise_ratio:.0%})"
    
    # 显示原始数据
    st.subheader("原始数据")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 特殊处理噪声点
    if -1 in y_true:
        # 绘制非噪声点
        mask_normal = y_true >= 0
        scatter = ax.scatter(X[mask_normal, 0], X[mask_normal, 1], 
                          c=y_true[mask_normal], cmap='viridis', s=50, alpha=0.7)
        # 绘制噪声点
        mask_noise = y_true == -1
        ax.scatter(X[mask_noise, 0], X[mask_noise, 1], color='gray', 
                 s=50, alpha=0.5, marker='x', label='噪声点')
        legend1 = ax.legend(*scatter.legend_elements(), title="真实簇")
        ax.add_artist(legend1)
        ax.legend()
    else:
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=50, alpha=0.7)
        legend = ax.legend(*scatter.legend_elements(), title="真实簇")
        ax.add_artist(legend)
    
    ax.set_title(title)
    ax.set_xlabel("特征1")
    ax.set_ylabel("特征2")
    st.pyplot(fig)
    
    # 聚类参数设置
    st.subheader("聚类算法参数")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("##### K-Means参数")
        k_value = st.slider("K值 (簇数量)", 2, 10, len(set(y_true)) - (1 if -1 in y_true else 0))
        kmeans_init = st.selectbox("初始化方法", ["k-means++", "random"])
        kmeans_n_init = st.slider("初始化次数", 1, 20, 10)
    
    with col2:
        st.write("##### DBSCAN参数")
        eps_value = st.slider("eps (邻域半径)", 0.1, 1.0, 0.3, 0.05)
        min_samples_value = st.slider("min_samples (最小点数)", 2, 20, 5)
    
    # 运行两种聚类算法
    kmeans = KMeans(
        n_clusters=k_value, 
        init=kmeans_init, 
        n_init=kmeans_n_init, 
        random_state=42
    )
    dbscan = DBSCAN(
        eps=eps_value, 
        min_samples=min_samples_value
    )
    
    kmeans_labels = kmeans.fit_predict(X)
    dbscan_labels = dbscan.fit_predict(X)
    
    # 计算评估指标
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        st.write("##### K-Means评估指标")
        
        # 轮廓系数
        try:
            silhouette_kmeans = silhouette_score(X, kmeans_labels)
            st.write(f"轮廓系数: {silhouette_kmeans:.4f}")
        except:
            st.write("轮廓系数: 无法计算")
        
        # 戴维斯-布尔丁指数
        try:
            dbi_kmeans = davies_bouldin_score(X, kmeans_labels)
            st.write(f"戴维斯-布尔丁指数: {dbi_kmeans:.4f}")
        except:
            st.write("戴维斯-布尔丁指数: 无法计算")
        
        # 与真实标签的一致性 (排除噪声点)
        if -1 in y_true:
            mask = y_true != -1
            if np.sum(mask) > 0:
                try:
                    ari_kmeans = adjusted_rand_score(y_true[mask], kmeans_labels[mask])
                    st.write(f"调整兰德指数 (不含噪声): {ari_kmeans:.4f}")
                except:
                    st.write("调整兰德指数: 无法计算")
        else:
            try:
                ari_kmeans = adjusted_rand_score(y_true, kmeans_labels)
                st.write(f"调整兰德指数: {ari_kmeans:.4f}")
            except:
                st.write("调整兰德指数: 无法计算")
    
    with metrics_col2:
        st.write("##### DBSCAN评估指标")
        
        # DBSCAN特有指标
        n_clusters_ = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise_ = list(dbscan_labels).count(-1)
        st.write(f"发现的簇数量: {n_clusters_}")
        st.write(f"噪声点数量: {n_noise_} ({n_noise_/len(dbscan_labels):.1%})")
        
        # 轮廓系数 (排除噪声点)
        if -1 in dbscan_labels and len(set(dbscan_labels) - {-1}) > 1:
            mask = dbscan_labels != -1
            if np.sum(mask) > 1:
                try:
                    silhouette_dbscan = silhouette_score(X[mask], dbscan_labels[mask])
                    st.write(f"轮廓系数 (不含噪声): {silhouette_dbscan:.4f}")
                except:
                    st.write("轮廓系数: 无法计算")
        elif len(set(dbscan_labels)) > 1:
            try:
                silhouette_dbscan = silhouette_score(X, dbscan_labels)
                st.write(f"轮廓系数: {silhouette_dbscan:.4f}")
            except:
                st.write("轮廓系数: 无法计算")
        else:
            st.write("轮廓系数: 无法计算 (簇数量不足)")
        
        # 与真实标签的一致性
        try:
            ari_dbscan = adjusted_rand_score(y_true, dbscan_labels)
            st.write(f"调整兰德指数: {ari_dbscan:.4f}")
        except:
            st.write("调整兰德指数: 无法计算")
    
    # 可视化聚类结果比较
    st.subheader("聚类结果比较")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # K-Means结果
    scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', s=50, alpha=0.7)
    centers = kmeans.cluster_centers_
    ax1.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='质心')
    ax1.set_title(f"K-Means聚类结果 (K={k_value})")
    ax1.set_xlabel("特征1")
    ax1.set_ylabel("特征2")
    legend1 = ax1.legend(*scatter1.legend_elements(), title="簇")
    ax1.add_artist(legend1)
    ax1.legend(loc='upper left')
    
    # DBSCAN结果
    # 为噪声点创建特殊颜色映射
    unique_labels = set(dbscan_labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels) - (1 if -1 in dbscan_labels else 0)))
    color_dict = {}
    
    i = 0
    for label in unique_labels:
        if label == -1:
            color_dict[label] = (0.7, 0.7, 0.7, 1.0)  # 灰色表示噪声
        else:
            color_dict[label] = colors[i]
            i += 1
    
    for label in unique_labels:
        mask = dbscan_labels == label
        if label == -1:
            ax2.scatter(X[mask, 0], X[mask, 1], s=50, color=color_dict[label], 
                      alpha=0.5, label='噪声点')
        else:
            ax2.scatter(X[mask, 0], X[mask, 1], s=50, color=color_dict[label], 
                      alpha=0.7, label=f'簇 {label}')
    
    ax2.set_title(f"DBSCAN聚类结果 (eps={eps_value}, min_samples={min_samples_value})")
    ax2.set_xlabel("特征1")
    ax2.set_ylabel("特征2")
    ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 算法对比小结
    st.subheader("对比小结")
    
    st.markdown("""
    ### K-Means vs. DBSCAN 主要区别
    
    | 特性 | K-Means | DBSCAN |
    | --- | --- | --- |
    | 簇形状 | 倾向于球状、凸形 | 可以是任意形状 |
    | 簇数量K | 需要预先指定 | 自动确定 |
    | 噪声/异常值 | 对异常值敏感，所有点都会被分配到某个簇 | 能识别噪声点，不将其分配到任何簇 |
    | 簇密度 | 对密度不敏感 | 对密度敏感，难以处理密度差异大的簇 |
    | 参数 | K | eps, min_samples |
    | 参数敏感度 | 对K和初始质心敏感 | 对eps和min_samples非常敏感 |
    | 计算复杂度 | 相对较低 | 可能较高（取决于数据和参数） |
    | 特征缩放 | 需要 | 需要 |
    """)
    
    # 降维可视化(高维数据聚类结果可视化)
    with st.expander("高维数据降维可视化 (示例)"):
        st.write("""
        对于高维数据，我们无法直接在二维平面上可视化聚类结果。通常使用降维技术如PCA或t-SNE将数据投影到低维空间进行可视化。
        
        下面展示了在MNIST数据集(手写数字图像, 784维)上的聚类结果，通过PCA和t-SNE降维到2维进行可视化。
        """)
        
        st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_kmeans_digits_001.png", 
                caption="MNIST数据PCA降维后的K-Means聚类结果")
        
        st.write("""
        **注意:**
        - **PCA**: 线性降维，快速但可能丢失非线性关系
        - **t-SNE**: 非线性降维，保留局部结构，但计算成本高，不适合大数据集
        - 降维可视化帮助直观理解聚类结果，但可能不完全反映高维空间中的真实簇结构
        """) 