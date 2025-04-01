import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.cluster import KMeans
from utils.font_manager import set_chinese_font

st.set_page_config(
    page_title="K-Means回顾",
    page_icon="🔄",
    layout="wide"
)

st.title("K-Means算法回顾与局限性")

st.markdown("""
我们在之前的课程中已经学习了K-Means算法，它是一种非常常用且高效的划分式聚类算法。

### 核心思想

将数据点划分为预先设定的 K 个簇，使得每个点都属于离其最近的簇中心（质心），并且簇内数据点的平方和最小。

### 算法步骤

1. 随机选择K个点作为初始簇中心
2. 将每个数据点分配到最近的簇中心
3. 重新计算每个簇的中心（均值）
4. 重复步骤2和3，直到簇中心基本不再变化或达到最大迭代次数

### 优点
""")

# 创建三列布局
col1, col2 = st.columns(2)

with col1:
    st.subheader("优点")
    st.markdown("""
    * **简单高效**：算法简单，易于理解和实现
    * **计算速度快**：适合处理大规模数据集
    * **适合发现球状簇**：当簇呈球形且分离明显时效果较好
    """)

with col2:
    st.subheader("缺点与局限性")
    st.markdown("""
    * **需要预先指定K值**：K值的选择对结果影响很大，选择不当会导致次优的聚类效果
    * **对初始质心敏感**：不同的初始质心可能导致不同的聚类结果
    * **对非球状簇效果不佳**：难以发现非规则形状的簇
    * **对异常值敏感**：异常值会对质心的计算产生较大影响
    """)

st.markdown("---")

st.subheader("缓解K-Means局限性的方法")
st.markdown("""
* **选择K值**：使用肘部法则、轮廓系数等方法辅助选择K值
* **初始化问题**：使用K-Means++进行初始化，或多次运行选择最佳结果（scikit-learn中的`n_init`参数）
* **非球状簇**：考虑使用其他聚类算法，如层次聚类、DBSCAN等
* **异常值**：预处理阶段进行异常值检测和处理

正是因为K-Means的这些局限性，我们需要学习其他聚类算法来应对更复杂的数据和场景。
""")

# 修改K-Means失效案例的部分，添加自己生成的图和更详细的解释
st.subheader("K-Means失效案例示例")

# 创建演示数据
set_chinese_font()  # 设置中文字体

# 生成半月形数据和K-Means结果
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
plt.subplots_adjust(hspace=0.4)

# 半月形数据
axs[0, 0].set_title('半月形数据 - 真实分布')
X_moons, y_moons = make_moons(n_samples=200, noise=0.05, random_state=42)
axs[0, 0].scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='viridis', s=40)
axs[0, 0].set_xlabel('特征1')
axs[0, 0].set_ylabel('特征2')

# 对半月形数据使用K-Means
kmeans_moons = KMeans(n_clusters=2, random_state=42)
y_kmeans_moons = kmeans_moons.fit_predict(X_moons)
axs[0, 1].set_title('半月形数据 - K-Means聚类结果')
axs[0, 1].scatter(X_moons[:, 0], X_moons[:, 1], c=y_kmeans_moons, cmap='viridis', s=40)
centers_moons = kmeans_moons.cluster_centers_
axs[0, 1].scatter(centers_moons[:, 0], centers_moons[:, 1], c='red', marker='X', s=100)
axs[0, 1].set_xlabel('特征1')
axs[0, 1].set_ylabel('特征2')

# 环形数据
axs[1, 0].set_title('环形数据 - 真实分布')
X_circles, y_circles = make_circles(n_samples=200, factor=0.5, noise=0.05, random_state=42)
axs[1, 0].scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap='viridis', s=40)
axs[1, 0].set_xlabel('特征1')
axs[1, 0].set_ylabel('特征2')

# 对环形数据使用K-Means
kmeans_circles = KMeans(n_clusters=2, random_state=42)
y_kmeans_circles = kmeans_circles.fit_predict(X_circles)
axs[1, 1].set_title('环形数据 - K-Means聚类结果')
axs[1, 1].scatter(X_circles[:, 0], X_circles[:, 1], c=y_kmeans_circles, cmap='viridis', s=40)
centers_circles = kmeans_circles.cluster_centers_
axs[1, 1].scatter(centers_circles[:, 0], centers_circles[:, 1], c='red', marker='X', s=100)
axs[1, 1].set_xlabel('特征1')
axs[1, 1].set_ylabel('特征2')

st.pyplot(fig)

st.markdown("""
**为什么K-Means在这些数据上失效？**

上图展示了K-Means在非球形数据上的明显局限性：

1. **半月形数据**：
   - 左图显示了两个半月形的真实簇
   - 右图是K-Means的聚类结果，可以看到它错误地将数据分割成了上下两部分
   - 原因：K-Means只能基于点到质心的欧氏距离进行划分，无法识别弯曲的结构

2. **环形数据**：
   - 左图显示了内圈和外圈两个环形簇
   - 右图是K-Means的聚类结果，它将数据错误地分成了左右两部分
   - 原因：K-Means假设簇是凸形的，无法处理环形或嵌套结构

**结论**：K-Means在处理非凸或复杂形状簇时效果不佳。对于这类数据结构，应考虑使用：
- DBSCAN算法（基于密度的聚类）
- 层次聚类（配合单连接可识别非凸形状）
- 谱聚类（处理复杂流形上的数据）
""")

# 添加页脚
st.markdown("---")
st.markdown("© 2024 机器学习课程 | 交互式聚类算法课件") 