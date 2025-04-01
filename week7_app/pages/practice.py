import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from utils.app_utils import create_custom_header, create_info_box, create_expander
from utils.data_generator import generate_blob_data, generate_moons_data, generate_circles_data, generate_anisotropic_data
from utils.visualization import plot_clusters

def show_practice():
    """显示实践环节页面"""
    create_custom_header("聚类分析实践环节", "应用聚类算法解决实际问题", "💻")
    
    # 创建标签页
    tab1, tab2, tab3 = st.tabs([
        "算法效果对比", 
        "数据探索实践", 
        "挑战与思考"
    ])
    
    with tab1:
        show_algorithm_comparison()
    
    with tab2:
        show_data_exploration()
    
    with tab3:
        show_challenges()

def show_algorithm_comparison():
    """不同聚类算法效果对比"""
    st.subheader("聚类算法效果对比")
    
    st.markdown("""
    在不同形状的数据集上，不同聚类算法的表现可能差异很大。本节我们将比较K-means、层次聚类和DBSCAN在各种形状数据上的表现。
    """)
    
    # 选择数据集类型
    dataset_type = st.selectbox(
        "选择数据集类型:",
        ["均匀分布的球状簇", "月牙形数据", "环形数据", "不均匀分布簇"],
        index=0
    )
    
    # 生成选定类型的数据
    if dataset_type == "均匀分布的球状簇":
        X, y_true = generate_blob_data(n_samples=300, n_centers=3, random_state=42)
        expected_clusters = 3
    elif dataset_type == "月牙形数据":
        X, y_true = generate_moons_data(n_samples=300, noise=0.1, random_state=42)
        expected_clusters = 2
    elif dataset_type == "环形数据":
        X, y_true = generate_circles_data(n_samples=300, noise=0.05, factor=0.5, random_state=42)
        expected_clusters = 2
    else:  # 不均匀分布
        X, y_true = generate_anisotropic_data(n_samples=300, n_centers=3, random_state=42)
        expected_clusters = 3
    
    # 绘制原始数据
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=50, alpha=0.8)
    ax.set_title("原始数据（真实标签）")
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)
    
    # 设置聚类参数
    n_clusters = st.slider("选择簇数量 (K):", 2, 6, expected_clusters)
    
    # 执行聚类算法
    # 1. K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    
    # 2. 层次聚类
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    hierarchical_labels = hierarchical.fit_predict(X)
    
    # 3. DBSCAN
    # 对于DBSCAN，我们需要设置eps和min_samples
    eps = st.slider("DBSCAN - 邻域半径 (eps):", 0.1, 2.0, 0.5, 0.1)
    min_samples = st.slider("DBSCAN - 最小样本数 (min_samples):", 2, 20, 5)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(X)
    
    # 计算轮廓系数 (对于DBSCAN，如果只有一个簇或全是噪声点，则无法计算)
    kmeans_silhouette = silhouette_score(X, kmeans_labels) if len(np.unique(kmeans_labels)) > 1 else np.nan
    hierarchical_silhouette = silhouette_score(X, hierarchical_labels) if len(np.unique(hierarchical_labels)) > 1 else np.nan
    
    unique_dbscan_labels = np.unique(dbscan_labels)
    if len(unique_dbscan_labels) > 1 and -1 not in unique_dbscan_labels:
        dbscan_silhouette = silhouette_score(X, dbscan_labels)
    elif len(unique_dbscan_labels) > 2 and -1 in unique_dbscan_labels:
        # 如果有噪声点但也有多个簇，我们可以只针对非噪声点计算
        mask = dbscan_labels != -1
        if np.sum(mask) > 0 and len(np.unique(dbscan_labels[mask])) > 1:
            dbscan_silhouette = silhouette_score(X[mask], dbscan_labels[mask])
        else:
            dbscan_silhouette = np.nan
    else:
        dbscan_silhouette = np.nan
    
    # 绘制聚类结果
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # K-means结果
    axes[0].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', s=50, alpha=0.8)
    kmeans_score = f"{kmeans_silhouette:.3f}" if not np.isnan(kmeans_silhouette) else "N/A"
    axes[0].set_title(f"K-means (K={n_clusters})\n轮廓系数: {kmeans_score}")
    axes[0].set_aspect('equal')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # 层次聚类结果
    axes[1].scatter(X[:, 0], X[:, 1], c=hierarchical_labels, cmap='viridis', s=50, alpha=0.8)
    hierarchical_score = f"{hierarchical_silhouette:.3f}" if not np.isnan(hierarchical_silhouette) else "N/A"
    axes[1].set_title(f"层次聚类 (K={n_clusters})\n轮廓系数: {hierarchical_score}")
    axes[1].set_aspect('equal')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # DBSCAN结果
    scatter = axes[2].scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis', s=50, alpha=0.8)
    dbscan_score = f"{dbscan_silhouette:.3f}" if not np.isnan(dbscan_silhouette) else "N/A"
    axes[2].set_title(f"DBSCAN (eps={eps}, min_samples={min_samples})\n轮廓系数: {dbscan_score}")
    axes[2].set_aspect('equal')
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    # 添加图例（只为DBSCAN添加，因为可能有噪声点）
    if -1 in dbscan_labels:
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                                    markersize=10, label='噪声点')]
        axes[2].legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 分析结果
    st.markdown("### 观察与分析")
    
    st.markdown(f"""
    根据上面的聚类结果，我们可以观察到：
    
    **K-means:**
    - 优点: 简单高效，对球形数据效果好
    - 限制: 对非球形数据表现较差，无法识别噪声点
    - 在当前数据上表现: {"良好" if kmeans_silhouette > 0.5 else "一般" if kmeans_silhouette > 0.3 else "较差"}
    
    **层次聚类:**
    - 优点: 可以处理各种形状的簇，提供聚类的层次结构
    - 限制: 计算复杂度高，不适合大型数据集
    - 在当前数据上表现: {"良好" if hierarchical_silhouette > 0.5 else "一般" if hierarchical_silhouette > 0.3 else "较差"}
    
    **DBSCAN:**
    - 优点: 能够发现任意形状的簇，自动识别噪声点
    - 限制: 参数敏感，对高维数据效果可能不佳
    - 在当前数据上表现: {"良好" if not np.isnan(dbscan_silhouette) and dbscan_silhouette > 0.5 else "一般" if not np.isnan(dbscan_silhouette) and dbscan_silhouette > 0.3 else "较差"}
    
    **总体分析:**
    - 对于{dataset_type}，{
        "K-means表现最佳" if kmeans_silhouette == max([kmeans_silhouette, hierarchical_silhouette, dbscan_silhouette if not np.isnan(dbscan_silhouette) else -float('inf')]) else 
        "层次聚类表现最佳" if hierarchical_silhouette == max([kmeans_silhouette, hierarchical_silhouette, dbscan_silhouette if not np.isnan(dbscan_silhouette) else -float('inf')]) else
        "DBSCAN表现最佳" if not np.isnan(dbscan_silhouette) and dbscan_silhouette == max([kmeans_silhouette, hierarchical_silhouette, dbscan_silhouette]) else
        "各算法表现相似"
    }
    """)
    
    # 提示用户尝试不同数据集和参数
    create_info_box("尝试不同数据集类型和参数设置，观察聚类算法在不同情况下的表现。特别关注算法对非球形数据的处理能力。", "info")

def show_data_exploration():
    """数据探索实践"""
    st.subheader("聚类分析实践：探索真实数据")
    
    st.markdown("""
    在这个环节中，我们将使用真实世界的数据集进行聚类分析，并探索如何解释聚类结果。
    """)
    
    # 加载演示数据
    st.markdown("### 数据：鸢尾花数据集（Iris Dataset）")
    
    st.markdown("""
    鸢尾花数据集包含三种不同种类鸢尾花的测量数据：
    - 花萼长度 (Sepal Length)
    - 花萼宽度 (Sepal Width)
    - 花瓣长度 (Petal Length)
    - 花瓣宽度 (Petal Width)
    
    我们将忽略真实标签，使用聚类算法来尝试发现数据中的自然分组。
    """)
    
    # 加载鸢尾花数据集
    from sklearn.datasets import load_iris
    
    iris = load_iris()
    X = iris.data
    y_true = iris.target
    feature_names = iris.feature_names
    
    # 显示数据集信息
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = [iris.target_names[i] for i in y_true]
    
    st.markdown("**数据集预览:**")
    st.dataframe(df.head())
    
    st.markdown(f"**数据形状:** {X.shape[0]} 行 × {X.shape[1]} 列")
    
    # 特征选择
    st.markdown("### 特征选择")
    
    st.markdown("""
    为了简化可视化，我们将选择两个特征进行聚类。请选择你想要使用的特征：
    """)
    
    feature1 = st.selectbox("特征 1:", feature_names, index=0)
    feature2 = st.selectbox("特征 2:", feature_names, index=2)
    
    # 提取所选特征
    X_selected = df[[feature1, feature2]].values
    
    # 数据预处理选项
    scale_data = st.checkbox("标准化数据 (均值=0, 标准差=1)", value=True)
    
    if scale_data:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
    else:
        X_scaled = X_selected
    
    # 绘制原始数据（带真实标签）
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_selected[:, 0], X_selected[:, 1], c=y_true, cmap='viridis', s=50, alpha=0.8)
    
    # 添加图例
    handles, labels = scatter.legend_elements()
    # 确保handles是普通Python列表而不是NumPy数组
    handles = list(handles)
    
    # 直接使用具名参数并传入Python原生类型
    ax.legend(handles=handles, 
             labels=list(iris.target_names),
             loc="upper right", 
             title="物种")
    
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.set_title("原始数据（真实标签）")
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)
    
    # 选择聚类算法
    algorithm = st.selectbox(
        "选择聚类算法:",
        ["K-means", "层次聚类 (Agglomerative)", "DBSCAN"],
        index=0
    )
    
    # 设置算法参数
    if algorithm == "K-means" or algorithm == "层次聚类 (Agglomerative)":
        n_clusters = st.slider("选择簇数量 (K):", 2, 8, 3)
    
    if algorithm == "DBSCAN":
        eps = st.slider("邻域半径 (eps):", 0.1, 2.0, 0.5, 0.1)
        min_samples = st.slider("最小样本数 (min_samples):", 2, 20, 5)
    
    # 执行聚类
    if algorithm == "K-means":
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif algorithm == "层次聚类 (Agglomerative)":
        linkage = st.selectbox("选择linkage方法:", ["ward", "complete", "average", "single"])
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    else:  # DBSCAN
        model = DBSCAN(eps=eps, min_samples=min_samples)
    
    y_pred = model.fit_predict(X_scaled)
    
    # 对于K-means，我们可以获取簇中心
    if algorithm == "K-means":
        # 如果数据被缩放，我们需要将簇中心转回原始尺度
        if scale_data:
            centers = scaler.inverse_transform(model.cluster_centers_)
        else:
            centers = model.cluster_centers_
    else:
        centers = None
    
    # 计算评估指标（如果可能）
    try:
        if len(np.unique(y_pred)) > 1 and -1 not in y_pred:
            silhouette = silhouette_score(X_scaled, y_pred)
            show_silhouette = True
        elif len(np.unique(y_pred)) > 2 and -1 in y_pred:
            # 如果有噪声点但也有多个簇，我们可以只针对非噪声点计算
            mask = y_pred != -1
            if np.sum(mask) > 1 and len(np.unique(y_pred[mask])) > 1:
                silhouette = silhouette_score(X_scaled[mask], y_pred[mask])
                show_silhouette = True
            else:
                show_silhouette = False
        else:
            show_silhouette = False
    except:
        show_silhouette = False
    
    # 绘制聚类结果
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制数据点
    scatter = ax.scatter(X_selected[:, 0], X_selected[:, 1], c=y_pred, cmap='viridis', s=50, alpha=0.8)
    
    # 绘制簇中心（仅对K-means）
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], s=200, marker='X', c='red', edgecolors='k', label='簇中心')
        ax.legend()
    
    # 为DBSCAN添加噪声点图例
    if algorithm == "DBSCAN" and -1 in y_pred:
        handles, labels = scatter.legend_elements()
        # 将NumPy数组转换为Python列表，避免布尔判断问题
        handles = list(handles)
        labels = list(labels)
        labels = [label if i != 0 else "噪声点" for i, label in enumerate(labels)]
        ax.legend(handles=handles, labels=labels, loc="upper right", title="簇")
    
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    alg_name = algorithm
    if algorithm == "K-means":
        alg_name += f" (K={n_clusters})"
    elif algorithm == "层次聚类 (Agglomerative)":
        alg_name += f" (K={n_clusters}, linkage={linkage})"
    else:  # DBSCAN
        alg_name += f" (eps={eps}, min_samples={min_samples})"
    
    if show_silhouette:
        ax.set_title(f"{alg_name}\n轮廓系数: {silhouette:.3f}")
    else:
        ax.set_title(alg_name)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)
    
    # 与真实标签比较
    st.markdown("### 聚类结果与真实标签比较")
    
    # 创建交叉表
    cross_tab = pd.crosstab(y_true, y_pred, 
                           rownames=['真实物种'], 
                           colnames=['预测簇'])
    
    # 添加行标签
    cross_tab.index = [iris.target_names[i] for i in range(len(iris.target_names))]
    
    st.table(cross_tab)
    
    # 计算外部评估指标
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
    
    # 对于包含噪声点的DBSCAN结果，我们只考虑被分配到簇的点
    if algorithm == "DBSCAN" and -1 in y_pred:
        mask = y_pred != -1
        if np.sum(mask) > 0:
            ari = adjusted_rand_score(y_true[mask], y_pred[mask])
            ami = adjusted_mutual_info_score(y_true[mask], y_pred[mask])
            noise_percentage = 100 * (1 - np.sum(mask) / len(mask))
        else:
            ari = ami = np.nan
            noise_percentage = 100
    else:
        ari = adjusted_rand_score(y_true, y_pred)
        ami = adjusted_mutual_info_score(y_true, y_pred)
        noise_percentage = 0
    
    cols = st.columns(3)
    
    with cols[0]:
        st.metric("调整兰德指数 (ARI)", f"{ari:.3f}" if not np.isnan(ari) else "N/A")
    
    with cols[1]:
        st.metric("调整互信息 (AMI)", f"{ami:.3f}" if not np.isnan(ami) else "N/A")
    
    if algorithm == "DBSCAN" and -1 in y_pred:
        with cols[2]:
            st.metric("噪声点比例", f"{noise_percentage:.1f}%")
    
    # 分析与解释
    st.markdown("### 聚类结果分析")
    
    # 提前计算所有复杂表达式，避免在f-string中出现复杂嵌套
    clusters_info = ""
    if algorithm != "K-means" and algorithm != "层次聚类 (Agglomerative)":
        cluster_count = len(np.unique(y_pred)) if -1 not in y_pred else len(np.unique(y_pred))-1
        clusters_info = f"对比真实标签（3种物种），算法识别出了{cluster_count}个簇"
    else:
        clusters_info = f"算法被设置为寻找{n_clusters}个簇"
    
    ari_value = f"{ari:.3f}" if not np.isnan(ari) else "无法计算"
    
    if not np.isnan(ari):
        if ari > 0.7:
            ari_quality = "(良好)"
        elif ari > 0.3:
            ari_quality = "(一般)"
        else:
            ari_quality = "(较差)"
    else:
        ari_quality = ""
    
    cluster_observation = ""
    if not np.isnan(ari) and ari > 0.3:
        cluster_observation = "聚类算法能够很好地区分Setosa物种，但对Versicolor和Virginica的区分较为困难"
    else:
        cluster_observation = "聚类效果与真实物种分类有较大差异"
    
    cluster_count = len(np.unique(y_pred)) if -1 not in y_pred else len(np.unique(y_pred))-1
    feature_observation = f"在所选的两个特征({feature1}和{feature2})空间中，数据呈现{cluster_count}个自然分组"
    
    feature_effect = ""
    if ari < 0.7 and not np.isnan(ari):
        feature_effect = f"特征选择影响了聚类效果 - {feature1}和{feature2}可能无法完全分离三个物种"
    else:
        feature_effect = "所选特征能够很好地表现数据的自然分组结构"
    
    algorithm_effect = ""
    if algorithm == "DBSCAN" or algorithm == "层次聚类 (Agglomerative)":
        algorithm_effect = "算法参数设置对结果有显著影响"
    else:
        algorithm_effect = "K的选择对K-means结果影响显著"
    
    # 使用更简单的f-string格式
    st.markdown(f"""
    **算法表现分析:**
    
    - 我们使用了{algorithm}算法对鸢尾花数据进行聚类
    - {clusters_info}
    - 与真实物种的匹配程度 (ARI): {ari_value} {ari_quality}
    
    **从聚类结果中能学到什么?**
    
    通过观察聚类结果和交叉表，我们可以发现:
    
    - {cluster_observation}
    - {feature_observation}
    
    **为什么会有这样的结果?**
    
    - {feature_effect}
    - {algorithm_effect}
    """)
    
    # 扩展:尝试其他特征组合
    st.markdown("### 进一步探索")
    
    st.markdown("""
    为了更全面地理解数据，尝试以下操作:
    
    1. 选择不同的特征组合，观察聚类效果变化
    2. 调整算法参数，寻找最佳聚类结果
    3. 比较不同聚类算法在相同特征上的表现差异
    """)

def show_challenges():
    """挑战与思考问题"""
    st.subheader("挑战与思考")
    
    st.markdown("""
    以下是一些关于聚类分析的进阶问题和挑战，供你思考：
    """)
    
    # 思考问题
    questions = [
        {
            "title": "高维空间的诅咒",
            "content": """
            随着数据维度的增加，聚类算法面临"高维空间的诅咒"问题：
            
            1. 距离度量变得不那么有意义，因为高维空间中点与点之间的距离趋于相等
            2. 数据变得更稀疏，难以形成密集簇
            
            **思考：** 如何在高维数据（如文本、基因表达数据）上有效应用聚类分析？
            
            **可能的解决方案包括：**
            - 降维技术（如PCA、t-SNE）
            - 特征选择
            - 子空间聚类方法
            """
        },
        {
            "title": "混合分布的聚类",
            "content": """
            在实际应用中，数据可能来自多个不同类型的分布：
            
            **思考：** 如何处理包含不同形状、大小和密度簇的数据？
            
            **尝试：** 
            - 组合多种聚类算法（如，先用DBSCAN识别密集区域，再对剩余点用K-means）
            - 考虑密度敏感的算法（如HDBSCAN、OPTICS）
            - 使用更灵活的模型（如混合模型）
            """
        },
        {
            "title": "大规模数据聚类",
            "content": """
            对于非常大型的数据集（百万到亿级数据点），传统聚类算法可能在计算上不可行。
            
            **思考：** 如何扩展聚类算法以处理大规模数据？
            
            **可能的方法：**
            - 采样技术（先在样本上聚类，再将结果推广）
            - 在线/流式聚类方法（单次扫描数据）
            - 分布式/并行聚类算法
            - 近似最近邻方法
            """
        },
        {
            "title": "自动参数选择",
            "content": """
            聚类算法通常需要手动指定关键参数（如K值、eps值等）。
            
            **挑战：** 如何自动选择最优参数而不依赖真实标签？
            
            **探索：**
            - 内部评估指标的稳定性分析
            - 模型稳定性方法（如bootstrap重采样）
            - 信息论方法（MDL原则等）
            - Gap统计量和类似方法
            """
        }
    ]
    
    # 显示思考问题
    for i, q in enumerate(questions):
        with st.expander(f"{i+1}. {q['title']}", expanded=i==0):
            st.markdown(q["content"])
    
    # 开放性挑战
    st.markdown("### 开放性挑战")
    
    st.markdown("""
    以下是一些可以尝试的开放性挑战：
    
    1. **无监督异常检测：** 使用聚类方法检测数据中的异常点，比较不同方法的有效性。
    
    2. **混合数据聚类：** 设计一种方法，能够有效地聚类包含数值和分类特征的混合数据。
    
    3. **动态数据聚类：** 探索如何对随时间变化的数据进行聚类，既要反映当前状态，又要考虑历史趋势。
    
    4. **视觉化分析：** 开发更直观的聚类结果可视化方法，尤其是对于高维数据。
    
    5. **领域应用：** 选择一个特定领域（如医疗健康、金融、社交网络等），应用聚类分析解决该领域的实际问题。
    """)
    
    # 资源推荐
    st.markdown("### 扩展学习资源")
    
    st.markdown("""
    如果你希望深入了解聚类分析，以下是一些推荐的学习资源：
    
    **书籍：**
    - *"数据挖掘：概念与技术"* (Jiawei Han, Micheline Kamber, Jian Pei)
    - *"Pattern Recognition and Machine Learning"* (Christopher Bishop)
    - *"Elements of Statistical Learning"* (Trevor Hastie, Robert Tibshirani, Jerome Friedman)
    
    **在线课程与教程：**
    - Scikit-learn聚类文档：[scikit-learn.org/stable/modules/clustering.html](https://scikit-learn.org/stable/modules/clustering.html)
    - Stanford CS246: 挖掘大规模数据集
    - Coursera上的数据挖掘专项课程
    
    **学术论文：**
    - *"A Survey of Clustering Data Mining Techniques"* (Pavel Berkhin)
    - *"Clustering"* (Jain et al., 1999)
    - *"A Density-Based Algorithm for Discovering Clusters"* (DBSCAN论文)
    
    **Python工具包：**
    - scikit-learn: 最常用的机器学习库
    - HDBSCAN: DBSCAN的层次化扩展
    - yellowbrick: 机器学习可视化
    """)
    
    create_info_box("记住：聚类分析是一门艺术，也是一门科学。技术工具很重要，但对业务领域的理解和批判性思维同样不可或缺。", "info") 