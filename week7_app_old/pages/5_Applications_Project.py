import streamlit as st

st.set_page_config(
    page_title="应用场景与项目",
    page_icon="💼",
    layout="wide"
)

st.title("层次聚类应用场景与小组项目三")

# 应用场景部分
st.header("层次聚类应用场景")

st.markdown("""
层次聚类由于其能够展示数据的层级结构，在许多领域都有广泛应用：
""")

# 创建四列布局展示各个应用场景
col1, col2 = st.columns(2)

with col1:
    st.subheader("市场细分 (Market Segmentation)")
    st.markdown("""
    * 通过对用户的购买行为、人口统计学特征、兴趣偏好等数据进行层次聚类
    * 发现不同层级的用户群体
    * 树状图可以直观地展示细分市场之间的关系和距离
    * 帮助营销人员理解市场的结构，制定更精准的营销策略
    * 例如，先区分高价值和低价值客户，再在高价值客户中细分出不同偏好的群体
    """)
    
    st.subheader("社交网络分析 (Social Network Analysis)")
    st.markdown("""
    * 社群发现：识别社交网络中的紧密连接的子群组或社区
    * 用户兴趣分类：根据用户的互动和内容偏好进行分层分类
    * 影响力分析：识别网络中具有不同层次影响力的用户群体
    """)

with col2:
    st.subheader("生物信息学 (Bioinformatics)")
    st.markdown("""
    * 基因表达谱分析：对不同样本或不同条件下的基因表达数据进行聚类
    * 发现功能相似或共同调控的基因群，或者对样本进行分类
    * 物种分类：构建物种的系统发育树
    * 蛋白质结构分析：根据结构相似性对蛋白质进行分类
    """)
    
    st.subheader("图像分割 (Image Segmentation)")
    st.markdown("""
    * 将图像中的像素根据颜色、纹理等特征进行层次聚类
    * 实现图像区域的划分
    * 可用于医学图像分析，例如识别MRI扫描中的不同组织
    * 用于计算机视觉中的物体识别和场景理解
    """)

# 小组项目三启动信息
st.header("小组项目三启动：用户分群模型构建与分析")

st.markdown("""
### 项目目标

掌握不同聚类算法的应用，理解聚类结果评估，并能对聚类结果进行业务解读。

### 核心要求

* **算法选择:** **至少尝试两种**聚类算法（从K-Means, 层次聚类, GMM, DBSCAN中选择）
* **对比分析:** 对比不同算法在你的数据集上的表现，解释选择最终模型的原因
  * 使用轮廓系数、Davies-Bouldin指数等指标进行评估
  * 结合可视化结果进行分析
* **结果解读:** 对最终的聚类结果进行业务层面的分析和解读
  * 例如，不同用户群体的特征是什么？
  * 这些特征对业务决策有何启示？

### 数据集选择

* 学生小组自主选择数据集
* 建议选择包含用户画像或用户行为特征的数据集
* 鼓励寻找能体现不同算法优势的数据
* 可选数据源：
  * [UCI机器学习库](https://archive.ics.uci.edu/ml/index.php)
  * [Kaggle数据集](https://www.kaggle.com/datasets)
  * 公司实习项目中的实际数据（注意隐私保护）
""")

# 项目时间线和可选挑战
project_col1, project_col2 = st.columns(2)

with project_col1:
    st.subheader("项目时间线")
    st.markdown("""
    * **初步模型与算法选择理由**: 第9周课前
      * 数据集选择和预处理
      * 初步尝试至少两种算法
      * 解释算法选择的理由
    
    * **最终模型、对比分析与报告**: 第10或11周课前（待定）
      * 完整的聚类模型实现
      * 不同算法的对比分析
      * 结果的业务解读
      * 项目演示和答辩
    """)

with project_col2:
    st.subheader("可选挑战")
    st.markdown("""
    鼓励有能力的小组尝试**基于文本Embeddings的聚类**（我们将在下周介绍概念）
    
    例如：
    * 对电商评论数据进行聚类
    * 对新闻文章进行主题聚类
    * 对客户反馈进行分类
    
    这将涉及：
    * 文本预处理
    * 使用预训练模型（如Word2Vec, BERT等）获取文本嵌入
    * 对嵌入向量进行聚类分析
    """)

# 项目评分标准
st.subheader("项目评分标准")
st.markdown("""
* **技术实现 (40%)**: 算法选择合理性、实现正确性、评估方法得当
* **对比分析 (30%)**: 不同算法的比较深度、选择依据的合理性
* **业务解读 (20%)**: 对聚类结果的解释深度、业务洞察的价值
* **演示与答辩 (10%)**: 演示清晰度、问题回答的准确性

**注意**: 优秀的项目不仅在技术上实现正确，更重要的是能从结果中提取有价值的业务洞察，帮助解决实际问题。
""")

# 添加页脚
st.markdown("---")
st.markdown("© 2024 机器学习课程 | 交互式聚类算法课件") 