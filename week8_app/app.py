import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from matplotlib.patches import Ellipse
# Check for sentence-transformers
try:
   from sentence_transformers import SentenceTransformer
   SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
   SENTENCE_TRANSFORMERS_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(
    page_title="第 8 周：聚类进阶与应用",
    page_icon="🧩",
    layout="wide"
)

# --- Sidebar Navigation ---
st.sidebar.title("导航")
page = st.sidebar.radio(
    "选择模块:",
    [
        "欢迎与介绍",
        "高斯混合模型 (GMM)",
        "DBSCAN",
        "聚类算法对比",
        "Embeddings 聚类初步",
        "思考与练习"
    ]
)

# --- Main Content Area ---

if page == "欢迎与介绍":
    st.title("第 8 周：概率聚类、密度聚类与现代应用 🧩")
    st.markdown("""
    欢迎来到第八周的学习！本周我们将深入探讨更高级的聚类技术，超越 K-Means 的局限性。

    **学习目标:**

    *   理解高斯混合模型 (GMM) 的原理、概率聚类思想和 EM 算法的基本概念。
    *   掌握使用 `scikit-learn` 实现 GMM 的方法。
    *   回顾 DBSCAN 算法的原理，并理解其在发现任意形状簇和异常点检测方面的优势。
    *   能够对比 K-Means、层次聚类、GMM 和 DBSCAN 的特点、适用场景和优缺点。
    *   理解基于 Embeddings 的聚类基本概念及其在处理文本、图像等非结构化数据中的应用。

    请使用左侧导航栏探索不同的模块。
    """)

elif page == "高斯混合模型 (GMM)":
    st.title("高斯混合模型 (Gaussian Mixture Models, GMM)")

    st.markdown("""
    我们在 K-Means 中假设簇是球状的，并且每个点只能硬性地属于一个簇。高斯混合模型 (GMM) 提供了一种更灵活的概率聚类方法。

    **核心思想:** GMM 假设数据是由 K 个不同的高斯分布（正态分布）混合生成的。每个高斯分布代表一个簇，具有自己的均值 (mean)、协方差 (covariance) 和权重 (weight)。

    **概率聚类 (Soft Clustering):** 与 K-Means 不同，GMM 计算的是每个数据点**属于每个高斯分布（簇）的概率**。一个点可以同时属于多个簇，只是概率不同。

    **参数:**
    *   **均值 (μ):** 每个高斯分布的中心。
    *   **协方差 (Σ):** 描述每个高斯分布的形状和方向 (圆形、椭圆形及方向)。
    *   **权重 (π):** 每个高斯分布在整个混合模型中所占的比例。

    **期望最大化 (Expectation-Maximization, EM) 算法:** GMM 通常使用 EM 算法来估计模型参数。这是一个迭代过程，交替进行 E 步（估计数据点属于各簇的概率）和 M 步（根据概率重新估计模型参数）。

    **优点:** 灵活性高 (拟合椭圆簇)、提供概率信息。
    **缺点:** 对初始化敏感、可能需要较多数据、计算可能较慢、需要预先指定 K 值。
    """)
    st.divider()

    st.subheader("交互式演示")

    # --- Data Generation Controls ---
    st.sidebar.header("GMM 数据生成")
    n_samples_gmm = st.sidebar.slider("样本数量 (N)", 100, 1000, 300, 50, key="gmm_n_samples")
    centers_gmm = st.sidebar.slider("簇中心数量 (K)", 2, 6, 3, 1, key="gmm_centers")
    cluster_std_gmm = st.sidebar.slider("簇标准差", 0.5, 3.0, 1.5, 0.1, key="gmm_std")
    random_state_gmm = st.sidebar.number_input("随机种子", value=170, key="gmm_seed")

    # Generate data
    X_gmm, y_true_gmm = make_blobs(n_samples=n_samples_gmm, centers=centers_gmm,
                                   cluster_std=cluster_std_gmm, random_state=random_state_gmm)
    # Apply a transformation to make blobs more elliptical for demonstration
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_gmm_transformed = np.dot(X_gmm, transformation)

    # --- GMM Parameter Controls ---
    st.sidebar.header("GMM 模型参数")
    n_components_gmm = st.sidebar.slider("GMM 组件数量 (K)", 1, 10, centers_gmm, 1, key="gmm_n_components")
    covariance_type_gmm = st.sidebar.selectbox(
        "协方差类型",
        ('full', 'tied', 'diag', 'spherical'),
        key="gmm_cov_type"
    )
    n_init_gmm = st.sidebar.slider("初始化次数 (n_init)", 1, 20, 10, 1, key="gmm_n_init")

    # --- GMM Fitting and Visualization ---
    try:
        # Fit GMM
        gmm = GaussianMixture(n_components=n_components_gmm,
                              covariance_type=covariance_type_gmm,
                              n_init=n_init_gmm,
                              random_state=42) # Use fixed random state for GMM fitting consistency
        gmm.fit(X_gmm_transformed)
        gmm_labels = gmm.predict(X_gmm_transformed)
        gmm_probs = gmm.predict_proba(X_gmm_transformed)

        # Visualization
        fig_gmm, ax_gmm = plt.subplots(figsize=(10, 8))

        # Scatter plot
        colors = plt.cm.viridis(gmm_labels / (n_components_gmm - 1)) if n_components_gmm > 1 else ['blue'] * len(X_gmm_transformed)
        ax_gmm.scatter(X_gmm_transformed[:, 0], X_gmm_transformed[:, 1], c=colors, s=10, alpha=0.7)

        # Plot Ellipses
        for i in range(n_components_gmm):
            mean = gmm.means_[i]
            if covariance_type_gmm == 'full':
                covar = gmm.covariances_[i]
            elif covariance_type_gmm == 'tied':
                covar = gmm.covariances_ # Shared covariance
            elif covariance_type_gmm == 'diag':
                covar = np.diag(gmm.covariances_[i])
            elif covariance_type_gmm == 'spherical':
                covar = np.eye(X_gmm_transformed.shape[1]) * gmm.covariances_[i]

            try:
                v, w = np.linalg.eigh(covar)
                v = 2. * np.sqrt(2.) * np.sqrt(v) # Scale ellipse to ~95% confidence
                u = w[0] / np.linalg.norm(w[0])
                angle = np.arctan2(u[1], u[0])
                angle = 180. * angle / np.pi # Convert to degrees
                ellipse = Ellipse(mean, v[0], v[1], angle=angle, fill=False, edgecolor='red', linewidth=2)
                ax_gmm.add_patch(ellipse)
            except np.linalg.LinAlgError:
                st.warning(f"无法为组件 {i} 绘制椭圆 (协方差矩阵可能是奇异的)。")


        ax_gmm.set_title(f'GMM 聚类 (k={n_components_gmm}, cov={covariance_type_gmm})')
        ax_gmm.set_xlabel('特征 1')
        ax_gmm.set_ylabel('特征 2')
        ax_gmm.grid(True)
        ax_gmm.axis('equal') # Ensure aspect ratio is equal for correct ellipse shape

        st.pyplot(fig_gmm)

        # Display probabilities (optional)
        if st.checkbox("显示前 5 个样本的概率", key="gmm_show_probs"):
            st.write("前 5 个样本属于每个簇的概率:")
            st.dataframe(gmm_probs[:5].round(3))

        # Display Silhouette Score
        if len(np.unique(gmm_labels)) > 1: # Silhouette score requires at least 2 labels
             score = silhouette_score(X_gmm_transformed, gmm_labels)
             st.metric("轮廓系数 (Silhouette Score)", f"{score:.3f}")
        else:
             st.info("轮廓系数需要至少 2 个簇才能计算。")

    except Exception as e:
        st.error(f"运行 GMM 时出错: {e}")
        st.error("请检查参数设置，特别是组件数量和协方差类型是否适合当前数据。")


elif page == "DBSCAN":
    st.title("DBSCAN (Density-Based Spatial Clustering of Applications with Noise)")
    st.markdown("""
    DBSCAN 是一种基于密度的聚类算法，擅长发现任意形状的簇并识别噪声点。

    **核心思想:** 寻找被低密度区域分隔的高密度区域作为簇。它不需要预先指定簇的数量。

    **关键概念:**
    *   **核心点 (Core Point):** 在半径 `eps` 内至少包含 `min_samples` 个点的点。
    *   **边界点 (Border Point):** 不是核心点，但在某个核心点的半径 `eps` 内。
    *   **噪声点 (Noise Point):** 既不是核心点也不是边界点。
    *   **密度可达/相连:** 定义了点如何基于密度形成簇。

    **算法流程:** 从一个未访问点开始，如果是核心点，则扩展其密度可达区域形成一个簇；如果不是，则暂时标记为噪声。

    **优点:** 可以发现任意形状的簇、对噪声点不敏感、无需预先指定 K 值。
    **缺点:** 对参数 `eps` 和 `min_samples` 敏感、对于密度变化较大的数据集效果不佳、高维数据效果可能下降。
    """)
    st.divider()

    st.subheader("交互式演示")

    # --- Dataset Selection ---
    st.sidebar.header("DBSCAN 数据集")
    dataset_options = {
        "两个月亮 (Moons)": "moons",
        "两个圆环 (Circles)": "circles",
        "斑点 (Blobs)": "blobs",
        "各向异性斑点 (Aniso)": "aniso",
        "不同方差斑点 (Varied)": "varied",
        "无结构 (No Structure)": "no_structure"
    }
    selected_dataset_name = st.sidebar.selectbox(
        "选择数据集",
        list(dataset_options.keys()),
        key="dbscan_dataset"
    )
    dataset_key = dataset_options[selected_dataset_name]

    n_samples_dbscan = st.sidebar.slider("样本数量 (N)", 100, 1500, 500, 50, key="dbscan_n_samples")
    noise_dbscan = st.sidebar.slider("噪声比例", 0.0, 0.2, 0.05, 0.01, key="dbscan_noise")
    random_state_dbscan = st.sidebar.number_input("随机种子", value=42, key="dbscan_seed")

    # Generate selected dataset
    X_dbscan = None
    y_true_dbscan = None # Not always available or relevant for DBSCAN visualization

    if dataset_key == "moons":
        X_dbscan, y_true_dbscan = make_moons(n_samples=n_samples_dbscan, noise=noise_dbscan, random_state=random_state_dbscan)
    elif dataset_key == "circles":
        X_dbscan, y_true_dbscan = make_circles(n_samples=n_samples_dbscan, factor=0.5, noise=noise_dbscan, random_state=random_state_dbscan)
    elif dataset_key == "blobs":
        X_dbscan, y_true_dbscan = make_blobs(n_samples=n_samples_dbscan, centers=3, cluster_std=0.8, random_state=random_state_dbscan)
        # Add noise manually for blobs if noise_dbscan > 0
        if noise_dbscan > 0:
             n_noise = int(n_samples_dbscan * noise_dbscan)
             # Adjust noise range based on typical blob data range
             blob_min, blob_max = X_dbscan.min(axis=0), X_dbscan.max(axis=0)
             noise_points = np.random.rand(n_noise, 2) * (blob_max - blob_min) + blob_min
             X_dbscan = np.vstack((X_dbscan, noise_points))
             y_true_dbscan = np.hstack((y_true_dbscan, [-1]*n_noise)) # Mark noise as -1 if needed
    elif dataset_key == "aniso":
        X_aniso, y_aniso = make_blobs(n_samples=n_samples_dbscan, random_state=random_state_dbscan, centers=3)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X_dbscan = np.dot(X_aniso, transformation)
        y_true_dbscan = y_aniso
    elif dataset_key == "varied":
        X_dbscan, y_true_dbscan = make_blobs(n_samples=n_samples_dbscan, cluster_std=[1.0, 2.5, 0.5], random_state=random_state_dbscan)
    elif dataset_key == "no_structure":
        X_dbscan = np.random.rand(n_samples_dbscan, 2) * 10
        y_true_dbscan = None # No true labels

    # --- DBSCAN Parameter Controls ---
    st.sidebar.header("DBSCAN 模型参数")
    # Adjust default eps based on dataset? Maybe too complex for now.
    eps_default = 0.3 if dataset_key in ["moons", "circles"] else 0.5
    eps = st.sidebar.slider("邻域半径 (eps)", 0.1, 2.0, eps_default, 0.05, key="dbscan_eps")
    min_samples = st.sidebar.slider("核心点最小样本数 (min_samples)", 2, 20, 5, 1, key="dbscan_min_samples")

    # --- DBSCAN Fitting and Visualization ---
    try:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(X_dbscan)
        dbscan_labels = dbscan.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise_ = list(dbscan_labels).count(-1)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("找到的簇数量", n_clusters_)
        with col2:
            st.metric("噪声点数量", n_noise_)


        # Visualization
        fig_dbscan, ax_dbscan = plt.subplots(figsize=(10, 8))

        unique_labels = set(dbscan_labels)
        # Generate colors, ensuring black is reserved for noise (-1)
        colors = {label: plt.cm.Spectral(each) for label, each in zip(unique_labels, np.linspace(0, 1, len(unique_labels))) if label != -1}
        colors[-1] = [0, 0, 0, 1] # Black for noise

        # Use a mask to separate core samples from noise
        core_samples_mask = np.zeros_like(dbscan_labels, dtype=bool)
        if hasattr(dbscan, 'core_sample_indices_') and len(dbscan.core_sample_indices_) > 0: # Check if core_sample_indices_ exists and is not empty
             core_samples_mask[dbscan.core_sample_indices_] = True


        for k in unique_labels:
            col = colors[k]
            class_member_mask = (dbscan_labels == k)

            # Plot core samples (slightly larger, less transparent)
            xy_core = X_dbscan[class_member_mask & core_samples_mask]
            if xy_core.shape[0] > 0:
                 ax_dbscan.plot(xy_core[:, 0], xy_core[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=8, alpha=0.8, label=f'Cluster {k}' if k != -1 else 'Noise (Core)')

            # Plot non-core samples (border points or noise points, smaller, more transparent)
            xy_non_core = X_dbscan[class_member_mask & ~core_samples_mask]
            if xy_non_core.shape[0] > 0:
                 marker = 'x' if k == -1 else 'o' # Use 'x' for noise points
                 markersize = 6 if k == -1 else 5
                 alpha = 0.6
                 label = 'Noise' if k == -1 and xy_core.shape[0] == 0 else None # Only label noise once
                 ax_dbscan.plot(xy_non_core[:, 0], xy_non_core[:, 1], marker, markerfacecolor=tuple(col), markeredgecolor='k', markersize=markersize, alpha=alpha, label=label)


        ax_dbscan.set_title(f'DBSCAN 聚类 (eps={eps:.2f}, min_samples={min_samples}) on {selected_dataset_name}')
        ax_dbscan.set_xlabel('特征 1')
        ax_dbscan.set_ylabel('特征 2')
        ax_dbscan.grid(True)
        # Add legend if clusters were found
        if n_clusters_ > 0:
             ax_dbscan.legend(loc='best')


        st.pyplot(fig_dbscan)

        # Display Silhouette Score (if possible)
        # Note: Silhouette score is not ideal for DBSCAN as it favors convex clusters,
        # but can still provide some comparative value.
        if n_clusters_ > 1:
            try:
                # Exclude noise points for silhouette calculation if desired
                # score_mask = dbscan_labels != -1
                # if np.sum(score_mask) > 1: # Need at least 2 non-noise points in >1 cluster
                #     score = silhouette_score(X_dbscan[score_mask], dbscan_labels[score_mask])
                #     st.metric("轮廓系数 (Silhouette Score - Excl. Noise)", f"{score:.3f}")
                # else:
                #     st.info("轮廓系数需要至少 2 个非噪声点才能计算。")

                # Calculate including noise points (as scikit-learn default would if labels were just 0, 1, ...)
                # This might be misleading for DBSCAN. Let's calculate on non-noise points.
                score_mask = dbscan_labels != -1
                labels_for_score = dbscan_labels[score_mask]
                if len(set(labels_for_score)) > 1: # Check if there are multiple clusters among non-noise points
                     score = silhouette_score(X_dbscan[score_mask], labels_for_score)
                     st.metric("轮廓系数 (Silhouette Score - Excl. Noise)", f"{score:.3f}")
                else:
                     st.info("轮廓系数需要至少 2 个非噪声簇才能计算。")

            except ValueError as e:
                 st.warning(f"计算轮廓系数时出错: {e}.")

        elif n_clusters_ == 1:
             st.info("只找到 1 个簇，无法计算轮廓系数。")
        else: # n_clusters_ == 0
             st.info("未找到簇 (所有点都是噪声)，无法计算轮廓系数。")


    except Exception as e:
        st.error(f"运行 DBSCAN 时出错: {e}")
        st.error("请检查参数设置。")


elif page == "聚类算法对比":
    st.title("聚类算法对比")

    st.markdown("""
    不同的聚类算法有不同的假设和优势。选择哪种算法取决于数据的特性和分析目标。下表总结了 K-Means、GMM 和 DBSCAN 的主要特点：
    """)

    # Display comparison table using Markdown
    comparison_data = """
    | 特性             | K-Means                     | GMM (高斯混合模型)          | DBSCAN                      |
    | :--------------- | :-------------------------- | :-------------------------- | :-------------------------- |
    | **簇形状**       | 球状 (各向同性)             | 椭圆状 (灵活)               | 任意形状                    |
    | **簇数量 (K)**   | 需预先指定                  | 需预先指定 (可用AIC/BIC)    | 无需预先指定                |
    | **聚类类型**     | 硬聚类                      | 软聚类 (概率)               | 硬聚类 (含噪声点)           |
    | **对异常值**     | 敏感                        | 相对鲁棒                    | 不敏感 (识别为噪声)         |
    | **主要优点**     | 简单、高效、适合球状簇      | 概率模型、适应椭圆簇        | 发现任意形状、处理噪声      |
    | **主要缺点**     | 对 K 和初始值敏感、限球状   | 对初始化敏感、计算可能较慢  | 对参数敏感、难处理密度变化 |
    | **适用场景**     | 簇较规则、数据量大          | 簇可能重叠或呈椭圆状        | 簇形状不规则、含噪声数据    |
    | **主要参数**     | `n_clusters`                | `n_components`, `covariance_type` | `eps`, `min_samples`        |
    """
    st.markdown(comparison_data)
    st.markdown("""
    *注：上表未包含层次聚类，其优点在于可无需预先指定 K 并可视化层次结构，但计算复杂度较高。*
    """)
    st.divider()

    st.subheader("交互式对比实验")

    # --- Dataset Selection (reuse DBSCAN's logic/controls) ---
    st.sidebar.header("对比实验数据集")
    dataset_options_comp = {
        "两个月亮 (Moons)": "moons",
        "两个圆环 (Circles)": "circles",
        "斑点 (Blobs)": "blobs",
        "各向异性斑点 (Aniso)": "aniso",
        "不同方差斑点 (Varied)": "varied",
        "无结构 (No Structure)": "no_structure"
    }
    selected_dataset_name_comp = st.sidebar.selectbox(
        "选择数据集",
        list(dataset_options_comp.keys()),
        key="comp_dataset"
    )
    dataset_key_comp = dataset_options_comp[selected_dataset_name_comp]

    n_samples_comp = st.sidebar.slider("样本数量 (N)", 100, 1500, 300, 50, key="comp_n_samples")
    noise_comp = st.sidebar.slider("噪声比例", 0.0, 0.2, 0.05, 0.01, key="comp_noise")
    random_state_comp = st.sidebar.number_input("随机种子", value=42, key="comp_seed")

    # Generate selected dataset
    X_comp = None
    y_true_comp = None

    # (Reusing dataset generation logic - consider refactoring to a helper function later)
    if dataset_key_comp == "moons":
        X_comp, y_true_comp = make_moons(n_samples=n_samples_comp, noise=noise_comp, random_state=random_state_comp)
    elif dataset_key_comp == "circles":
        X_comp, y_true_comp = make_circles(n_samples=n_samples_comp, factor=0.5, noise=noise_comp, random_state=random_state_comp)
    elif dataset_key_comp == "blobs":
        X_comp, y_true_comp = make_blobs(n_samples=n_samples_comp, centers=3, cluster_std=0.8, random_state=random_state_comp)
        if noise_comp > 0:
             n_noise = int(n_samples_comp * noise_comp)
             blob_min, blob_max = X_comp.min(axis=0), X_comp.max(axis=0)
             noise_points = np.random.rand(n_noise, 2) * (blob_max - blob_min) + blob_min
             X_comp = np.vstack((X_comp, noise_points))
             y_true_comp = np.hstack((y_true_comp, [-1]*n_noise))
    elif dataset_key_comp == "aniso":
        X_aniso, y_aniso = make_blobs(n_samples=n_samples_comp, random_state=random_state_comp, centers=3)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X_comp = np.dot(X_aniso, transformation)
        y_true_comp = y_aniso
    elif dataset_key_comp == "varied":
        X_comp, y_true_comp = make_blobs(n_samples=n_samples_comp, cluster_std=[1.0, 2.5, 0.5], random_state=random_state_comp)
    elif dataset_key_comp == "no_structure":
        X_comp = np.random.rand(n_samples_comp, 2) * 10
        y_true_comp = None

    # --- Algorithm Parameter Controls ---
    st.sidebar.header("算法参数")
    # K-Means
    k_kmeans = st.sidebar.slider("K-Means: K 值", 1, 10, 3, 1, key="comp_kmeans_k")
    n_init_kmeans = st.sidebar.slider("K-Means: 初始化次数 (n_init)", 1, 20, 10, 1, key="comp_kmeans_n_init")
    # GMM
    k_gmm = st.sidebar.slider("GMM: 组件数量 (K)", 1, 10, 3, 1, key="comp_gmm_k")
    cov_type_gmm = st.sidebar.selectbox("GMM: 协方差类型", ('full', 'tied', 'diag', 'spherical'), key="comp_gmm_cov")
    n_init_gmm_comp = st.sidebar.slider("GMM: 初始化次数 (n_init)", 1, 20, 10, 1, key="comp_gmm_n_init")
    # DBSCAN
    eps_dbscan = st.sidebar.slider("DBSCAN: eps", 0.1, 2.0, 0.3, 0.05, key="comp_dbscan_eps")
    min_samples_dbscan = st.sidebar.slider("DBSCAN: min_samples", 2, 20, 5, 1, key="comp_dbscan_min")

    # --- Run Algorithms and Display Results ---
    col1, col2, col3 = st.columns(3)

    # --- K-Means ---
    with col1:
        st.subheader("K-Means")
        try:
            kmeans = KMeans(n_clusters=k_kmeans, n_init=n_init_kmeans, random_state=random_state_comp)
            kmeans.fit(X_comp)
            kmeans_labels = kmeans.labels_
            centers = kmeans.cluster_centers_

            fig_kmeans, ax_kmeans = plt.subplots(figsize=(6, 5))
            colors_kmeans = plt.cm.viridis(kmeans_labels / k_kmeans) if k_kmeans > 0 else ['blue'] * len(X_comp)
            ax_kmeans.scatter(X_comp[:, 0], X_comp[:, 1], c=colors_kmeans, s=10, alpha=0.7)
            if k_kmeans > 0:
                ax_kmeans.scatter(centers[:, 0], centers[:, 1], marker='X', s=100, c='red', label='Centroids')
            ax_kmeans.set_title(f'K-Means (k={k_kmeans})')
            ax_kmeans.set_xticks(())
            ax_kmeans.set_yticks(())
            ax_kmeans.grid(True)
            st.pyplot(fig_kmeans)

            if len(np.unique(kmeans_labels)) > 1:
                score_kmeans = silhouette_score(X_comp, kmeans_labels)
                st.metric("轮廓系数", f"{score_kmeans:.3f}")
            else:
                st.info("轮廓系数需至少 2 个簇")
        except Exception as e:
            st.error(f"K-Means 运行出错: {e}")

    # --- GMM ---
    with col2:
        st.subheader("GMM")
        try:
            gmm_comp = GaussianMixture(n_components=k_gmm, covariance_type=cov_type_gmm, n_init=n_init_gmm_comp, random_state=random_state_comp)
            gmm_comp.fit(X_comp)
            gmm_labels_comp = gmm_comp.predict(X_comp)

            fig_gmm_comp, ax_gmm_comp = plt.subplots(figsize=(6, 5))
            colors_gmm = plt.cm.viridis(gmm_labels_comp / k_gmm) if k_gmm > 0 else ['blue'] * len(X_comp)
            ax_gmm_comp.scatter(X_comp[:, 0], X_comp[:, 1], c=colors_gmm, s=10, alpha=0.7)

            # Plot Ellipses (simplified for comparison view)
            for i in range(k_gmm):
                 mean = gmm_comp.means_[i]
                 try:
                     if cov_type_gmm == 'full': covar = gmm_comp.covariances_[i]
                     elif cov_type_gmm == 'tied': covar = gmm_comp.covariances_
                     elif cov_type_gmm == 'diag': covar = np.diag(gmm_comp.covariances_[i])
                     else: covar = np.eye(X_comp.shape[1]) * gmm_comp.covariances_[i]

                     v, w = np.linalg.eigh(covar)
                     v = 2. * np.sqrt(2.) * np.sqrt(v) # ~95% confidence
                     u = w[0] / np.linalg.norm(w[0])
                     angle = 180. * np.arctan2(u[1], u[0]) / np.pi
                     ellipse = Ellipse(mean, v[0], v[1], angle=angle, fill=False, edgecolor='red', lw=1)
                     ax_gmm_comp.add_patch(ellipse)
                 except np.linalg.LinAlgError:
                     pass # Ignore if ellipse cannot be drawn

            ax_gmm_comp.set_title(f'GMM (k={k_gmm}, cov={cov_type_gmm})')
            ax_gmm_comp.set_xticks(())
            ax_gmm_comp.set_yticks(())
            ax_gmm_comp.grid(True)
            st.pyplot(fig_gmm_comp)

            if len(np.unique(gmm_labels_comp)) > 1:
                score_gmm = silhouette_score(X_comp, gmm_labels_comp)
                st.metric("轮廓系数", f"{score_gmm:.3f}")
            else:
                st.info("轮廓系数需至少 2 个簇")
        except Exception as e:
            st.error(f"GMM 运行出错: {e}")


    # --- DBSCAN ---
    with col3:
        st.subheader("DBSCAN")
        try:
            dbscan_comp = DBSCAN(eps=eps_dbscan, min_samples=min_samples_dbscan)
            dbscan_comp.fit(X_comp)
            dbscan_labels_comp = dbscan_comp.labels_
            n_clusters_dbscan = len(set(dbscan_labels_comp)) - (1 if -1 in dbscan_labels_comp else 0)
            n_noise_dbscan = list(dbscan_labels_comp).count(-1)

            fig_dbscan_comp, ax_dbscan_comp = plt.subplots(figsize=(6, 5))
            unique_labels_db = set(dbscan_labels_comp)
            colors_db = {label: plt.cm.Spectral(each) for label, each in zip(unique_labels_db, np.linspace(0, 1, len(unique_labels_db))) if label != -1}
            colors_db[-1] = [0, 0, 0, 1] # Black for noise

            for k in unique_labels_db:
                col = colors_db[k]
                class_mask = (dbscan_labels_comp == k)
                marker = 'x' if k == -1 else 'o'
                markersize = 4 if k == -1 else 6
                alpha = 0.6 if k == -1 else 0.8
                ax_dbscan_comp.plot(X_comp[class_mask, 0], X_comp[class_mask, 1], marker,
                                    markerfacecolor=tuple(col), markeredgecolor='k',
                                    markersize=markersize, alpha=alpha)

            ax_dbscan_comp.set_title(f'DBSCAN (eps={eps_dbscan:.2f}, min={min_samples_dbscan})')
            ax_dbscan_comp.set_xticks(())
            ax_dbscan_comp.set_yticks(())
            ax_dbscan_comp.grid(True)
            st.pyplot(fig_dbscan_comp)

            st.write(f"找到 {n_clusters_dbscan} 个簇, {n_noise_dbscan} 个噪声点")
            # Calculate silhouette score excluding noise
            score_mask_db = dbscan_labels_comp != -1
            labels_for_score_db = dbscan_labels_comp[score_mask_db]
            if len(set(labels_for_score_db)) > 1:
                score_dbscan = silhouette_score(X_comp[score_mask_db], labels_for_score_db)
                st.metric("轮廓系数 (Excl. Noise)", f"{score_dbscan:.3f}")
            else:
                st.info("轮廓系数需至少 2 个非噪声簇")

        except Exception as e:
            st.error(f"DBSCAN 运行出错: {e}")


elif page == "Embeddings 聚类初步":
    st.title("新概念：基于 Embeddings 的聚类")

    st.markdown("""
    传统的聚类算法主要处理数值型数据。但现实中我们经常遇到**非结构化数据**，如文本、图像、音频等。如何对这些数据进行聚类？

    **核心思想:** 将高维、稀疏、非结构化的数据，通过 **Embedding 技术** 转换为低维、稠密的**向量表示 (Embeddings)**，这些向量能够捕捉原始数据的语义信息。然后，在这些 Embedding 向量上应用**标准的聚类算法**（如 K-Means, DBSCAN 等）。

    *   **什么是 Embedding?**
        *   一种 "编码" 或 "映射"，将复杂对象（如一个词、一篇文档、一张图片）表示为一个固定长度的实数向量。
        *   好的 Embedding 应该使得语义上相似的对象在向量空间中距离也相近。
    *   **常见的 Embedding 技术:**
        *   **文本:** Word2Vec, GloVe, FastText, Sentence-BERT (SBERT), Universal Sentence Encoder (USE), BERT, GPT 等。
        *   **图像:** 通过预训练的 CNN (ResNet, VGG 等) 提取特征向量。
    *   **流程:**
        1.  **选择/训练 Embedding 模型:** 根据数据类型选择合适的模型。
        2.  **数据转换:** 将原始数据输入 Embedding 模型，得到向量表示。
        3.  **聚类:** 在得到的 Embedding 向量上应用 K-Means, DBSCAN 等算法。
        4.  **结果分析:** 分析聚类结果，解读每个簇的含义。
    """)
    st.divider()

    st.subheader("简化文本聚类演示")

    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        st.warning("""
        **依赖库缺失:** `sentence-transformers` 未安装。

        请在您的终端运行以下命令来安装它：
        ```bash
        pip install sentence-transformers
        ```
        安装后请重新运行 Streamlit 应用。
        """)
    else:
        # --- Text Input ---
        default_text = """
        这只猫非常可爱。
        今天天气真好，适合散步。
        我喜欢吃披萨和汉堡。
        小狗在公园里跑来跑去。
        阳光明媚，万里无云。
        这家餐厅的意大利面很棒。
        那只宠物狗很活泼。
        """
        input_texts_str = st.text_area(
            "输入文本 (每行一个文档/句子):",
            value=default_text,
            height=200,
            key="embed_text_input"
        )
        input_texts = [text.strip() for text in input_texts_str.strip().split('\n') if text.strip()]

        if not input_texts:
            st.warning("请输入至少一行文本。")
        else:
            # --- Embedding Model Selection (Simplified) ---
            # Using a lightweight pre-trained model
            model_name = 'paraphrase-MiniLM-L6-v2'
            try:
                # Use caching to avoid reloading the model on every interaction
                @st.cache_resource
                def load_embedding_model(name):
                    return SentenceTransformer(name)

                model = load_embedding_model(model_name)
                st.info(f"使用 Sentence Transformer 模型: `{model_name}` (首次加载可能需要时间下载)")

                # --- Generate Embeddings ---
                with st.spinner("正在生成文本 Embeddings..."):
                    embeddings = model.encode(input_texts)
                st.write(f"已生成 {embeddings.shape[0]} 个 Embeddings，每个维度为 {embeddings.shape[1]}")

                # --- Clustering Algorithm Selection ---
                st.sidebar.header("文本聚类参数")
                cluster_algo = st.sidebar.selectbox(
                    "选择聚类算法",
                    ["K-Means", "DBSCAN"],
                    key="embed_algo"
                )

                cluster_labels = None
                n_clusters_found = 0

                if cluster_algo == "K-Means":
                    k_embed = st.sidebar.slider("K-Means: K 值", 1, max(1, len(input_texts) // 2), 2, 1, key="embed_kmeans_k")
                    n_init_embed = st.sidebar.slider("K-Means: 初始化次数", 1, 10, 5, 1, key="embed_kmeans_ninit")
                    try:
                        kmeans_embed = KMeans(n_clusters=k_embed, n_init=n_init_embed, random_state=42)
                        kmeans_embed.fit(embeddings)
                        cluster_labels = kmeans_embed.labels_
                        n_clusters_found = k_embed
                    except Exception as e:
                        st.error(f"K-Means 运行出错: {e}")

                elif cluster_algo == "DBSCAN":
                    eps_embed = st.sidebar.slider("DBSCAN: eps", 0.1, 5.0, 1.0, 0.1, key="embed_dbscan_eps")
                    min_samples_embed = st.sidebar.slider("DBSCAN: min_samples", 1, 10, 2, 1, key="embed_dbscan_min")
                    try:
                        dbscan_embed = DBSCAN(eps=eps_embed, min_samples=min_samples_embed, metric='cosine') # Cosine distance often works well for embeddings
                        dbscan_embed.fit(embeddings)
                        cluster_labels = dbscan_embed.labels_
                        n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                    except Exception as e:
                        st.error(f"DBSCAN 运行出错: {e}")

                # --- Display Results ---
                if cluster_labels is not None:
                    st.subheader("聚类结果")
                    results_data = {"文本": input_texts, "簇标签": cluster_labels}
                    st.dataframe(results_data)

                    # Optionally display silhouette score if applicable
                    if n_clusters_found > 1:
                         try:
                             # Exclude noise for DBSCAN score
                             score_mask_embed = cluster_labels != -1 if cluster_algo == "DBSCAN" else np.ones_like(cluster_labels, dtype=bool)
                             labels_for_score_embed = cluster_labels[score_mask_embed]
                             if len(set(labels_for_score_embed)) > 1:
                                 score_embed = silhouette_score(embeddings[score_mask_embed], labels_for_score_embed, metric='cosine' if cluster_algo == "DBSCAN" else 'euclidean')
                                 st.metric(f"轮廓系数 ({'Excl. Noise' if cluster_algo == 'DBSCAN' else ''})", f"{score_embed:.3f}")
                             else:
                                 st.info("轮廓系数需至少 2 个 (非噪声) 簇")
                         except ValueError as e:
                             st.warning(f"计算轮廓系数时出错: {e}")
                    elif n_clusters_found == 1:
                         st.info("只找到 1 个簇，无法计算轮廓系数。")
                    else: # 0 clusters found (all noise in DBSCAN)
                         st.info("未找到簇，无法计算轮廓系数。")


            except Exception as e:
                st.error(f"加载 Embedding 模型或生成 Embeddings 时出错: {e}")
                st.error("请确保网络连接正常，并且依赖库已正确安装。")


elif page == "思考与练习":
    st.title("思考与练习")
    st.markdown("点击下面的问题展开查看：")

    with st.expander("1. GMM 与 K-Means 的主要区别是什么？GMM 的“软聚类”体现在哪里？"):
        st.markdown("""
        *   **主要区别:**
            *   **簇形状假设:** K-Means 假设簇是球状且大小相似的 (基于最小化平方误差和)，而 GMM 假设簇是由高斯分布生成的，可以拟合椭圆形的簇 (基于最大化似然)。
            *   **分配方式:** K-Means 是硬聚类 (每个点属于一个簇)，GMM 是软聚类 (每个点以一定概率属于每个簇)。
            *   **模型基础:** K-Means 是基于距离的启发式算法，GMM 是基于概率模型的算法 (通常用 EM 优化)。
        *   **软聚类体现:** GMM 不直接分配标签，而是计算每个数据点属于每个高斯分量（簇）的后验概率。`predict_proba()` 方法返回的就是这些概率值，表示了点属于每个簇的不确定性。最终的硬分配标签 (`predict()`) 通常是选择概率最高的那个簇。
        """)

    with st.expander("2. DBSCAN 相比于 K-Means 的主要优势是什么？它在什么场景下特别有用？"):
        st.markdown("""
        *   **主要优势:**
            *   **发现任意形状的簇:** K-Means 难以处理非球状簇，而 DBSCAN 基于密度，可以发现任意形状的簇 (如月亮形、环形)。
            *   **处理噪声点:** K-Means 会强制将所有点分配到某个簇，对异常值敏感；DBSCAN 可以识别并标记出噪声点，对异常值不敏感。
            *   **无需预先指定 K 值:** K-Means 需要预先设定簇的数量 K，而 DBSCAN 的簇数量由算法根据数据密度和参数 (`eps`, `min_samples`) 自动确定。
        *   **特别有用的场景:**
            *   数据集包含非球状、形状不规则的簇。
            *   数据集中可能含有噪声或异常点需要识别和排除。
            *   簇的数量未知或难以预先估计。
            *   需要基于局部密度进行聚类的场景。
        """)

    with st.expander("3. 假设你要对一个包含不同密度区域和一些噪声点的数据集进行聚类，你会优先考虑哪种算法？为什么？"):
        st.markdown("""
        *   **优先考虑 DBSCAN。**
        *   **原因:**
            *   **处理噪声点:** DBSCAN 明确设计用来识别和处理噪声点，这符合数据集的特性。
            *   **处理不同密度区域:** 虽然标准的 DBSCAN 对全局参数 `eps` 和 `min_samples` 敏感，可能难以同时处理密度差异很大的簇，但它仍然是处理密度问题的基础算法。相比之下，K-Means 和 GMM 对密度变化更不适应。
            *   **任意形状:** 如果不同密度的区域还伴随着不规则形状，DBSCAN 的优势更加明显。
        *   **注意:** 如果密度差异**非常**大，可能需要考虑 DBSCAN 的变种，如 OPTICS 或 HDBSCAN*，它们能更好地处理密度变化。但作为优先考虑，DBSCAN 是最直接的选择。
        """)

    with st.expander("4. 解释“基于 Embeddings 的聚类”的基本流程。为什么需要 Embedding 这一步？"):
        st.markdown("""
        *   **基本流程:**
            1.  **选择/训练 Embedding 模型:** 根据数据类型（文本、图像等）选择合适的预训练模型 (如 Sentence-BERT, ResNet) 或自行训练模型。
            2.  **数据转换:** 将原始的非结构化数据（如句子列表、图片集合）输入 Embedding 模型，得到每个数据点对应的低维、稠密的向量表示 (Embeddings)。
            3.  **聚类:** 在生成的 Embedding 向量上应用标准的聚类算法 (如 K-Means, DBSCAN, GMM)。这些算法可以直接处理数值型向量。
            4.  **结果分析:** 分析聚类结果，例如查看同一簇内的文本/图像，理解簇的语义含义。
        *   **为什么需要 Embedding:**
            *   **处理非结构化数据:** 传统的聚类算法（如 K-Means, DBSCAN）通常需要数值型输入。文本、图像等是非结构化的，不能直接输入这些算法。
            *   **捕捉语义信息:** Embedding 技术可以将非结构化数据转换为能够捕捉其内在语义信息的向量。例如，语义相似的句子或内容相似的图片在 Embedding 空间中的向量距离会更近。
            *   **降维与特征提取:** Embedding 通常将高维、稀疏的数据（如词袋模型表示的文本）转换为低维、稠密的向量，更适合聚类算法处理，并能提取出更有效的特征。
        """)

    with st.expander("5. 如果你要对大量用户评论进行聚类以发现主要抱怨点，你会如何设计技术方案？"):
        st.markdown("""
        *   **技术方案设计:**
            1.  **数据预处理:** 清洗评论文本，例如去除无关字符、表情符号、HTML 标签，可能进行小写转换、分词等（取决于后续 Embedding 模型的要求）。
            2.  **选择 Embedding 模型:**
                *   优先考虑 **Sentence-BERT (SBERT)** 或类似的句子/段落 Embedding 模型。它们专门为捕捉句子级别的语义相似度而设计，效果通常优于简单的词向量平均。可以选择针对中文预训练的模型。
                *   如果评论较长，也可以考虑使用 BERT 等模型提取 [CLS] token 或对词向量进行池化。
            3.  **生成 Embeddings:** 将预处理后的评论输入选定的 Embedding 模型，为每条评论生成一个向量。
            4.  **选择聚类算法:**
                *   **DBSCAN:** 优点是不需要预设 K 值，可以自动发现簇的数量，并且能识别出一些独特的、不成簇的评论作为噪声。缺点是需要调整 `eps` 和 `min_samples` 参数，可能对参数敏感。使用 `cosine` 距离度量通常在 Embedding 空间效果较好。
                *   **K-Means:** 优点是简单快速。缺点是需要预先指定 K 值（抱怨点的类别数量），并且假设簇是球状的，这在语义空间中不一定成立。可以尝试不同的 K 值，并结合轮廓系数等指标评估。
                *   **GMM:** 也可以尝试，但可能计算成本较高，且同样需要预设 K 值。
                *   **HDBSCAN\*:** (如果可用) 是 DBSCAN 的改进，对参数选择更鲁棒，能处理不同密度的簇，可能是个不错的选择。
            5.  **执行聚类:** 在评论的 Embedding 向量上运行选定的聚类算法。
            6.  **结果分析与解读:**
                *   对于每个簇，提取代表性的评论或关键词（例如使用 TF-IDF 分析簇内文本）。
                *   人工检查每个簇的评论，归纳总结该簇代表的主要抱怨点主题。
                *   分析噪声点（如果使用 DBSCAN），看看是否代表一些独特的或不常见的抱怨。
                *   可视化（如果可能，例如使用 UMAP 或 t-SNE 将 Embeddings 降维到二维后绘制散点图）。
            7.  **迭代优化:** 根据聚类结果和业务理解，调整 Embedding 模型、聚类算法或其参数，重新进行聚类，直到获得满意的、可解释的结果。
        """)


# --- Helper Functions (to be added later) ---

# Example helper function structure
# def plot_gmm_results(X, labels, means, covariances, n_components, ax):
#     # ... plotting logic ...
#     pass