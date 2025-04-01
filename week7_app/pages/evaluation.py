import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, AgglomerativeClustering

from utils.app_utils import create_custom_header, create_info_box, render_latex, create_expander
from utils.data_generator import generate_blob_data, generate_anisotropic_data
from utils.visualization import plot_clusters, plot_silhouette

def show_evaluation():
    """显示聚类评估页面"""
    create_custom_header("聚类评估与算法选择", "如何评价聚类结果的好坏？", "📊")
    
    # 创建标签页
    tab1, tab2, tab3, tab4 = st.tabs([
        "评估指标概述", 
        "内部评估指标", 
        "外部评估指标", 
        "算法选择"
    ])
    
    with tab1:
        show_evaluation_overview()
    
    with tab2:
        show_internal_metrics()
    
    with tab3:
        show_external_metrics()
    
    with tab4:
        show_algorithm_selection()

def show_evaluation_overview():
    """聚类评估指标概述"""
    st.subheader("为什么需要聚类评估？")
    
    st.markdown("""
    评估聚类结果的质量对于确保我们的聚类算法有效是至关重要的。但聚类评估面临一个本质困难：
    **聚类是无监督学习，我们通常没有"标准答案"来比较**。
    
    主要面临的挑战包括：
    
    * **簇的数量选择：** 应该将数据分成多少簇？
    * **算法选择：** 哪种聚类算法最适合我们的数据？
    * **参数设置：** 如何为所选算法选择最佳参数？
    * **结果有效性：** 我们的聚类结果是否有意义？
    """)
    
    # 两大类评估方法
    st.markdown("### 两大类评估方法")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **内部评估 (Internal Evaluation):**
        
        * 仅使用数据本身和聚类结果
        * 不需要外部标签
        * 通常基于簇内紧密性和簇间分离性
        * 例如：轮廓系数、Davies-Bouldin 指数等
        
        *当没有外部标准时使用*
        """)
    
    with col2:
        st.markdown("""
        **外部评估 (External Evaluation):**
        
        * 将聚类结果与外部提供的标签对比
        * 需要已知的标签或分类
        * 度量聚类与真实标签的匹配程度
        * 例如：Rand 指数、调整兰德指数、互信息等
        
        *当有真实标签可参考时使用*
        """)
    
    # 评估的关键原则
    st.subheader("评估的关键原则")
    
    st.markdown("""
    无论使用哪种评估方法，好的聚类结果通常应满足两个关键原则：
    
    1. **紧凑性 (Compactness):** 簇内的点应该彼此接近（簇内距离小）
    2. **分离性 (Separation):** 不同簇之间应该有明显区分（簇间距离大）
    
    不同的评估指标以不同的方式结合这两个原则。
    """)
    
    # 示意图
    st.markdown("### 紧凑性与分离性示意图")
    
    # 生成一些示例数据
    np.random.seed(42)
    # 案例1：良好的簇（既紧凑又分离）
    X_good = np.vstack([
        np.random.randn(50, 2) * 0.5 + np.array([0, 0]),
        np.random.randn(50, 2) * 0.5 + np.array([5, 0])
    ])
    labels_good = np.hstack([np.zeros(50), np.ones(50)])
    
    # 案例2：不紧凑的簇
    X_not_compact = np.vstack([
        np.random.randn(50, 2) * 2.0 + np.array([0, 0]),
        np.random.randn(50, 2) * 2.0 + np.array([5, 0])
    ])
    labels_not_compact = np.hstack([np.zeros(50), np.ones(50)])
    
    # 案例3：未分离的簇
    X_not_separated = np.vstack([
        np.random.randn(50, 2) * 0.5 + np.array([0, 0]),
        np.random.randn(50, 2) * 0.5 + np.array([2, 0])
    ])
    labels_not_separated = np.hstack([np.zeros(50), np.ones(50)])
    
    # 绘制图表
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].scatter(X_good[:, 0], X_good[:, 1], c=labels_good, cmap='viridis', s=50, alpha=0.8)
    axs[0].set_title("良好的聚类\n(高紧凑性，高分离性)")
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    axs[1].scatter(X_not_compact[:, 0], X_not_compact[:, 1], c=labels_not_compact, cmap='viridis', s=50, alpha=0.8)
    axs[1].set_title("低紧凑性聚类\n(簇内距离大)")
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    axs[2].scatter(X_not_separated[:, 0], X_not_separated[:, 1], c=labels_not_separated, cmap='viridis', s=50, alpha=0.8)
    axs[2].set_title("低分离性聚类\n(簇间距离小)")
    axs[2].grid(True, linestyle='--', alpha=0.7)
    
    for ax in axs:
        ax.set_aspect('equal')
    
    plt.tight_layout()
    st.pyplot(fig)

def show_internal_metrics():
    """内部评估指标"""
    st.subheader("内部评估指标")
    
    st.markdown("""
    内部评估指标不依赖于外部标签，仅基于簇内紧密性和簇间分离性来评估聚类质量。
    """)
    
    # 常用内部评估指标
    metrics = {
        "轮廓系数 (Silhouette Coefficient)": {
            "描述": "结合了簇内紧凑性和簇间分离性的度量，值范围为 [-1, 1]。",
            "公式": r"s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}",
            "解释": "其中 a(i) 是数据点 i 与同簇其他点的平均距离（紧凑性），b(i) 是数据点 i 与最近的非同簇点的平均距离（分离性）。",
            "优良值": "接近 1 表示聚类效果好，接近 -1 表示效果差。",
            "特点": "对于凸形且大小相近的簇效果好，对于细长形状或密度不均的簇可能不适用。"
        },
        "Davies-Bouldin 指数 (Davies-Bouldin Index)": {
            "描述": "评估簇内分散程度与簇间距离的比率，值越小越好。",
            "公式": r"DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \left( \frac{\sigma_i + \sigma_j}{d(c_i, c_j)} \right)",
            "解释": "其中 σᵢ 是簇 i 内点到质心的平均距离，d(cᵢ, cⱼ) 是簇 i 和簇 j 质心之间的距离，k 是簇数量。",
            "优良值": "值越小越好，没有上限。",
            "特点": "特别适合评估簇的紧凑性和分离性平衡，但对噪声较敏感。"
        },
        "Calinski-Harabasz 指数 (Calinski-Harabasz Index)": {
            "描述": "也称为方差比标准（Variance Ratio Criterion），比较簇间离差与簇内离差，值越大越好。",
            "公式": r"CH = \frac{SS_B}{SS_W} \times \frac{N-k}{k-1}",
            "解释": "其中 SS_B 是簇间平方和（簇间分离性），SS_W 是簇内平方和（簇内紧凑性），N 是数据点数，k 是簇数量。",
            "优良值": "值越大越好，没有上限。", 
            "特点": "适合评估球状簇，对离群点敏感。"
        }
    }
    
    # 选择指标查看详情
    metric = st.selectbox(
        "选择指标查看详情:",
        list(metrics.keys())
    )
    
    # 显示指标详情
    metric_info = metrics[metric]
    st.markdown(f"### {metric}")
    st.markdown(f"**描述:** {metric_info['描述']}")
    
    st.markdown("**数学表达式:**")
    render_latex(metric_info["公式"])
    
    st.markdown(f"**解释:** {metric_info['解释']}")
    st.markdown(f"**优良值:** {metric_info['优良值']}")
    st.markdown(f"**特点:** {metric_info['特点']}")
    
    # 通过实例演示轮廓系数
    if metric == "轮廓系数 (Silhouette Coefficient)":
        st.markdown("### 轮廓系数可视化演示")
        
        # 生成数据及聚类
        X, y = generate_blob_data(n_samples=300, n_centers=3, random_state=42)
        
        # 用户选择簇数
        n_clusters = st.slider("选择簇数量:", 2, 6, 3)
        
        # 执行K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # 计算轮廓系数
        silhouette_avg = silhouette_score(X, cluster_labels)
        
        # 绘制轮廓图
        fig = plot_silhouette(X, cluster_labels)
        st.pyplot(fig)
        
        # 描述解读
        st.markdown(f"""
        **解读轮廓系数图:**
        
        * 平均轮廓系数值为 {silhouette_avg:.3f}。
        * 每个簇的轮廓分析显示在左图中，宽度表示该簇中样本数量。
        * 垂直红色虚线表示平均轮廓系数。
        * 任何簇的轮廓系数低于平均值或接近零/负值都表示聚类结果可能不理想。
        """)
    
    # 对比不同指标
    st.markdown("### 不同指标在各种聚类场景下的表现")
    
    # 生成数据
    data_option = st.radio(
        "选择数据类型:",
        ["均匀分布的球状簇", "不均匀分布的簇"],
        horizontal=True
    )
    
    if data_option == "均匀分布的球状簇":
        X, y_true = generate_blob_data(n_samples=300, n_centers=3, random_state=42)
    else:
        X, y_true = generate_anisotropic_data(n_samples=300, n_centers=3, random_state=42)
    
    # 绘制原始数据
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=50, alpha=0.8)
    ax.set_title("原始数据点")
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)
    
    # 测试不同的簇数量
    k_range = range(2, 11)
    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_harabasz_scores = []
    
    for k in k_range:
        # K-means聚类
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        
        # 计算内部评估指标
        silhouette_scores.append(silhouette_score(X, labels))
        davies_bouldin_scores.append(davies_bouldin_score(X, labels))
        calinski_harabasz_scores.append(calinski_harabasz_score(X, labels))
    
    # 绘制各指标随簇数量的变化
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].plot(k_range, silhouette_scores, 'o-', color='blue')
    axs[0].set_title('轮廓系数 (值越大越好)')
    axs[0].set_xlabel('簇数量 (k)')
    axs[0].set_xticks(k_range)
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    axs[1].plot(k_range, davies_bouldin_scores, 'o-', color='red')
    axs[1].set_title('Davies-Bouldin 指数 (值越小越好)')
    axs[1].set_xlabel('簇数量 (k)')
    axs[1].set_xticks(k_range)
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    axs[2].plot(k_range, calinski_harabasz_scores, 'o-', color='green')
    axs[2].set_title('Calinski-Harabasz 指数 (值越大越好)')
    axs[2].set_xlabel('簇数量 (k)')
    axs[2].set_xticks(k_range)
    axs[2].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 最优簇数建议
    best_k_silhouette = k_range[np.argmax(silhouette_scores)]
    best_k_db = k_range[np.argmin(davies_bouldin_scores)]
    best_k_ch = k_range[np.argmax(calinski_harabasz_scores)]
    
    st.markdown(f"""
    **指标建议的最优簇数量:**
    
    * 轮廓系数: **k = {best_k_silhouette}** (最大值: {max(silhouette_scores):.3f})
    * Davies-Bouldin 指数: **k = {best_k_db}** (最小值: {min(davies_bouldin_scores):.3f})
    * Calinski-Harabasz 指数: **k = {best_k_ch}** (最大值: {max(calinski_harabasz_scores):.0f})
    
    *注: 不同指标可能给出不同的建议，这是因为它们评估聚类质量的角度不同。*
    """)

def show_external_metrics():
    """外部评估指标"""
    st.subheader("外部评估指标")
    
    st.markdown("""
    外部评估指标通过将聚类结果与"真实"标签（外部提供的分组信息）进行比较来评估聚类质量。
    这些指标在有标签可用时特别有用，例如在算法研发阶段或基准测试数据集上。
    """)
    
    # 常用外部评估指标
    metrics = {
        "兰德指数 (Rand Index)": {
            "描述": "衡量聚类结果与真实标签之间的相似度，值范围为 [0, 1]。",
            "公式": r"RI = \frac{a + b}{a + b + c + d} = \frac{a + b}{\binom{n}{2}}",
            "解释": "其中 a 是同一真实簇且被分到同一预测簇的点对数量，b 是不同真实簇且被分到不同预测簇的点对数量，c、d 表示不一致的点对，n 是数据点数量。",
            "优良值": "接近 1 表示聚类与真实标签高度匹配。",
            "特点": "直观易懂，但对大量数据计算复杂度高，且不调整随机聚类。"
        },
        "调整兰德指数 (Adjusted Rand Index, ARI)": {
            "描述": "兰德指数的调整版，考虑了随机分配的影响，值范围为 [-1, 1]。",
            "公式": r"ARI = \frac{RI - E[RI]}{\max(RI) - E[RI]}",
            "解释": "其中 E[RI] 是随机分配的期望兰德指数，调整后使得随机分配的期望值为 0。",
            "优良值": "接近 1 表示完美匹配，0 表示随机分配，负值表示比随机更差。",
            "特点": "对随机性进行了校正，更适合作为聚类评估指标。"
        },
        "互信息 (Mutual Information, MI)": {
            "描述": "衡量真实分布与预测分布之间共享的信息量。",
            "公式": r"MI(U, V) = \sum_{i=1}^{|U|} \sum_{j=1}^{|V|} \frac{|U_i \cap V_j|}{N} \log \frac{N |U_i \cap V_j|}{|U_i| |V_j|}",
            "解释": "其中 U 和 V 分别是真实簇和预测簇的集合，|Uᵢ ∩ Vⱼ| 是同时属于真实簇 i 和预测簇 j 的点数量，N 是总点数。",
            "优良值": "值越大越好，没有上限。",
            "特点": "对簇的大小和数量敏感，因此通常使用其归一化或调整版本。"
        },
        "调整互信息 (Adjusted Mutual Information, AMI)": {
            "描述": "互信息的调整版，考虑了随机聚类的影响，值范围为 [0, 1]。",
            "公式": r"AMI(U, V) = \frac{MI(U, V) - E[MI(U, V)]}{\max(H(U), H(V)) - E[MI(U, V)]}",
            "解释": "其中 E[MI] 是随机分配的期望互信息，H(U) 和 H(V) 分别是真实分布和预测分布的熵。",
            "优良值": "接近 1 表示高度匹配，0 表示随机分配。",
            "特点": "对簇的数量不太敏感，适合比较不同算法和参数。"
        }
    }
    
    # 选择指标查看详情
    metric = st.selectbox(
        "选择指标查看详情:",
        list(metrics.keys())
    )
    
    # 显示指标详情
    metric_info = metrics[metric]
    st.markdown(f"### {metric}")
    st.markdown(f"**描述:** {metric_info['描述']}")
    
    st.markdown("**数学表达式:**")
    render_latex(metric_info["公式"])
    
    st.markdown(f"**解释:** {metric_info['解释']}")
    st.markdown(f"**优良值:** {metric_info['优良值']}")
    st.markdown(f"**特点:** {metric_info['特点']}")
    
    # 外部评估指标的注意事项
    st.markdown("""
    ### 使用外部评估指标的注意事项
    
    1. **标签对齐问题：** 聚类算法产生的簇标签与真实标签的具体数值可能不同（例如，算法可能将类别1标为簇2），但这不影响聚类的质量评估，因为外部指标仅关注样本是否被分到同一组。
    
    2. **标签不平衡：** 当真实标签的类别分布不平衡时，某些指标可能会给出误导性结果，应选择对类别平衡不敏感的指标（如ARI或AMI）。
    
    3. **实际应用中的限制：** 在实际应用中，我们通常没有真实标签（否则就不需要聚类了），因此外部评估指标主要用于算法开发和基准测试阶段。
    """)
    
    # 简单的外部评估指标示例计算
    st.markdown("### 示例：计算调整兰德指数 (ARI)")
    
    # 生成数据
    X, y_true = generate_blob_data(n_samples=300, n_centers=3, random_state=42)
    
    # 执行KMeans聚类
    k_slider = st.slider("选择簇数量 (K):", 2, 6, 3)
    
    # 计算聚类结果
    kmeans = KMeans(n_clusters=k_slider, random_state=42)
    y_pred = kmeans.fit_predict(X)
    
    # 可视化原始标签和聚类结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制真实标签
    ax1.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=50, alpha=0.8)
    ax1.set_title("真实标签")
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制聚类结果
    ax2.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50, alpha=0.8)
    ax2.set_title(f"KMeans聚类结果 (K={k_slider})")
    ax2.set_aspect('equal')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 计算外部评估指标
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
    
    ari = adjusted_rand_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    
    st.markdown(f"""
    **外部评估指标结果:**
    
    * 调整兰德指数 (ARI): **{ari:.3f}**
    * 调整互信息 (AMI): **{ami:.3f}**
    
    **解读:**
    
    * ARI 和 AMI 的值越接近 1，表示聚类结果越接近真实标签。
    * 当 K={k_slider} 时，聚类质量为 {ari:.3f}（根据ARI），这表明聚类结果与真实标签的{ari>=0.7 and "一致性较高" or "一致性一般"}。
    """)

def show_algorithm_selection():
    """算法选择方法"""
    st.subheader("如何选择合适的聚类算法？")
    
    st.markdown("""
    选择合适的聚类算法是聚类分析中的关键步骤，取决于多种因素，包括数据特性、分析目标和计算资源等。
    """)
    
    # 聚类算法对比
    st.markdown("### 主要聚类算法比较")
    
    algorithm_comparison = {
        "算法": ["K-means", "层次聚类 (凝聚型)", "DBSCAN", "高斯混合模型 (GMM)", "谱聚类 (Spectral Clustering)"],
        "适用簇形状": ["球形", "任意形状", "任意密度和形状", "椭圆形", "非凸形状"],
        "簇数量": ["需预先指定", "可从树状图选择", "自动确定", "需预先指定", "需预先指定"],
        "数据规模": ["大型", "小到中型", "大型", "中型", "小到中型"],
        "复杂度": ["O(nkdi)", "O(n²logn) ~ O(n³)", "O(n²) 最坏情况", "O(nkd²i)", "O(n³)"],
        "异常值敏感性": ["高", "方法依赖", "低 (视为噪声)", "中等", "低"],
        "特点": [
            "简单高效，结果易解释",
            "提供数据层次结构",
            "能发现任意形状簇，自动识别噪声",
            "软聚类，提供点属于各簇的概率",
            "能处理复杂形状，基于图论"
        ]
    }
    
    # 显示比较表格
    st.table(pd.DataFrame(algorithm_comparison))
    
    # 算法选择流程图（文字版）
    st.markdown("### 算法选择流程")
    
    st.markdown("""
    以下是一个简化的聚类算法选择流程：
    
    1. **数据规模如何？**
       - 非常大 → 考虑 K-means 或 DBSCAN
       - 中小型 → 可考虑所有算法
    
    2. **簇的形状是什么？**
       - 预期是球形或椭圆形 → K-means 或 GMM
       - 可能是任意形状 → 层次聚类、DBSCAN 或谱聚类
    
    3. **是否知道簇的数量？**
       - 知道确切数量 → K-means 或 GMM
       - 大致知道范围 → 层次聚类或谱聚类
       - 完全不知道 → DBSCAN 或层次聚类 + 树状图分析
    
    4. **数据中是否有噪声/离群点？**
       - 有显著噪声 → DBSCAN（自动检测噪声）或稳健版 K-means
       - 噪声较少 → 任何算法都可考虑
    
    5. **是否需要层次结构？**
       - 需要层次关系 → 层次聚类
       - 不需要 → 其他算法
    
    6. **计算资源是否有限？**
       - 非常有限 → K-means
       - 适中 → 根据其他因素选择
    """)
    
    # 案例分析
    st.markdown("### 案例分析：不同情境下的算法选择")
    
    case_studies = {
        "案例": [
            "客户细分 (50,000条记录)", 
            "基因表达分析 (500个样本)", 
            "图像分割", 
            "社交网络社区发现"
        ],
        "数据特点": [
            "高维属性，可能有噪声，预期形成几个主要客户群", 
            "高维数据，需要找出表达模式相似的基因组", 
            "像素点，需要识别图像中的不同对象", 
            "图结构数据，社区结构复杂"
        ],
        "推荐算法": [
            "K-means (快速处理大数据) 或 GMM (提供概率归属)", 
            "层次聚类 (提供层次关系视图)", 
            "DBSCAN (处理复杂形状) 或 谱聚类 (注重边界)", 
            "谱聚类 (基于图的算法) 或 社区检测算法"
        ],
        "原因": [
            "高效处理大量记录，客户群通常分布合理，适合K-means；若需软聚类则用GMM", 
            "样本量适中，层次聚类可提供多尺度视图，有助发现基因表达模式间的嵌套关系", 
            "图像中的对象可能形状复杂，需要能够处理任意形状的算法", 
            "社交网络中的社区结构通常是复杂的非欧氏关系，需要能捕捉这种复杂性的算法"
        ]
    }
    
    # 显示案例分析表格
    st.table(pd.DataFrame(case_studies))
    
    # 评估与选择总结
    st.markdown("""
    ### 聚类评估与算法选择的总结建议
    
    1. **综合使用多种评估指标：** 不同指标反映聚类质量的不同方面，综合考虑会更全面。
    
    2. **考虑领域知识：** 任何指标都无法替代对数据和业务的深入理解，最终结果应该是"有意义的"。
    
    3. **可视化验证：** 对于低维数据，可视化聚类结果是最直观的评估方法。
    
    4. **尝试多种算法：** 不同算法可能揭示数据的不同方面，值得比较多种结果。
    
    5. **迭代改进：** 聚类分析通常是探索性的，需要多次尝试不同参数和算法以获得最佳结果。
    
    6. **注意数据预处理：** 特征缩放、降维等预处理步骤往往对聚类结果有重大影响。
    """)
    
    create_info_box("记住：聚类分析通常是探索性的，而非确定性的。它提供的是数据的一种可能解释，而非绝对真理。", "info") 