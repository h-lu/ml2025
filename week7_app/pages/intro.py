import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.app_utils import create_custom_header, create_expander, create_info_box, create_quiz

def show_intro():
    """显示课程目标页面"""
    create_custom_header("课程目标", "掌握聚类分析的核心概念与方法", "🎯")
    
    st.markdown("""
    本节课结束时，学生将能够：
    """)
    
    # 使用st.columns创建两列布局
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        * **深入理解**聚类分析的核心概念、目标和多样化的应用场景。
        * **熟练掌握** K-means 聚类算法的原理、数学表达、步骤，并能分析其对初始化的敏感性及 K 值选择策略。
        * **清晰理解**凝聚型层次聚类的原理、不同 Linkage 方法的计算方式及其对结果的影响，并能解读树状图。
        """)
    
    with col2:
        st.markdown("""
        * **掌握**常用的聚类评估指标（内部与外部）及其应用场景。
        * **能够**根据数据特点和分析目标，选择合适的聚类算法。
        * **初步了解**在 Python 中实现基本聚类算法的方法。
        * **培养**对算法局限性的批判性思维和解决实际问题的能力。
        """)
    
    # 添加互动自评问卷
    st.subheader("课前自评")
    
    # 创建三列布局
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.slider("对聚类分析的了解程度", 1, 5, 1, help="1=完全不了解, 5=非常了解")
    
    with col2:
        st.slider("对K-means算法的了解程度", 1, 5, 1, help="1=完全不了解, 5=非常了解")
    
    with col3:
        st.slider("对层次聚类的了解程度", 1, 5, 1, help="1=完全不了解, 5=非常了解")
    
    # 添加本节课主要内容预览
    st.subheader("本节课主要内容")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["聚类分析导论", "K-means聚类", "层次聚类", "聚类评估", "实践应用"])
    
    with tab1:
        st.markdown("""
        * 什么是聚类？—— 无监督的探索
        * 聚类 vs. 分类：目标与方法的差异
        * 生活中的聚类应用场景
        """)
    
    with tab2:
        st.markdown("""
        * 算法核心思想与步骤
        * 互动环节：手动 K-means 模拟
        * 关键参数 K 的选择：肘部法则
        * K-means 的优缺点
        """)
    
    with tab3:
        st.markdown("""
        * 核心思想：合并或分裂
        * 凝聚型层次聚类的步骤与树状图
        * Linkage 方法：定义簇间距离
        * 层次聚类的优缺点
        """)
    
    with tab4:
        st.markdown("""
        * 如何评价聚类结果的好坏？
        * 内部评估指标 vs. 外部评估指标
        * 轮廓系数、DB指数等评估方法
        """)
    
    with tab5:
        st.markdown("""
        * 如何选择合适的聚类算法？
        * Python 实现基础
        * 案例分析
        """)
    
    # 提示信息
    create_info_box("课程结束后，您将能够理解聚类分析的原理，掌握K-means和层次聚类的基本方法，并能够选择合适的算法解决实际问题。")

def show_clustering_intro():
    """显示聚类分析导论页面"""
    create_custom_header("聚类分析：发现数据中的自然分组", "无监督学习的重要方法", "🔍")
    
    st.subheader("什么是聚类？—— 无监督的探索")
    
    # 核心思想
    st.markdown("""
    **核心思想：** 聚类是一种无监督学习技术，其目标是在没有预先定义标签的情况下，发现数据中隐藏的、自然的群组结构。
    我们希望组内（簇内）的数据点彼此相似，而组间（簇间）的数据点差异较大。
    """)
    
    # 可视化示例
    st.subheader("可视化示例")
    # 生成数据
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(50, 2) * 0.5 + np.array([2, 2]),  # 簇1
        np.random.randn(50, 2) * 0.5 + np.array([-2, 2]),  # 簇2
        np.random.randn(50, 2) * 0.5 + np.array([0, -2])   # 簇3
    ])
    
    # 绘制散点图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X[:, 0], X[:, 1], s=50, alpha=0.8, edgecolors='w')
    ax.set_title("数据点散点图（未标记）")
    ax.set_xlabel("特征 1")
    ax.set_ylabel("特征 2")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_aspect('equal')
    st.pyplot(fig)
    
    # 提问
    st.markdown("**思考：** 从这张图中，你能看出数据大概可以分成几组吗？你是如何判断的？")
    
    # 交互式选择答案
    clusters = st.radio("你认为数据应该分成几组？", [2, 3, 4, 5], horizontal=True)
    
    if st.button("查看分组结果"):
        # 根据用户选择显示不同的聚类结果
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8, edgecolors='w')
        ax.scatter(centroids[:, 0], centroids[:, 1], s=200, marker='X', c='red', edgecolors='k', label='质心')
        
        legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="簇")
        ax.add_artist(legend1)
        ax.legend()
        
        ax.set_title(f"聚类结果 (K={clusters})")
        ax.set_xlabel("特征 1")
        ax.set_ylabel("特征 2")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_aspect('equal')
        
        st.pyplot(fig)
        
        if clusters == 3:
            st.success("很好！数据确实可以自然地分为3组。")
        else:
            st.info(f"你选择了将数据分为{clusters}组。由于聚类是无监督的，没有绝对正确的答案，不过这组数据的自然分组看起来是3组。")
    
    # 生活中的聚类应用
    st.subheader("生活中的聚类")
    
    # 使用表格展示应用场景
    applications = {
        "领域": ["零售", "社交媒体", "计算机视觉", "生物信息学", "客户关系管理", "推荐系统"],
        "应用": [
            "购物篮分析，发现哪些商品经常被一起购买",
            "识别具有相似兴趣的用户群体",
            "图像分割，将图像中的像素分组",
            "发现具有相似表达模式的基因",
            "客户分群，根据消费行为将客户分类",
            "基于相似用户的内容推荐"
        ],
        "目的": [
            "优化货架摆放或制定捆绑销售策略",
            "社区发现或精准广告投放",
            "物体识别或背景分离",
            "研究基因功能联系",
            "针对性营销策略制定",
            "提高推荐精度"
        ]
    }
    
    df = pd.DataFrame(applications)
    st.table(df)
    
    # 互动思考
    st.markdown("**互动思考：** 您能想到其他聚类应用场景吗？")
    user_application = st.text_input("请输入您想到的聚类应用场景")
    
    if user_application:
        st.success(f'非常好！"{user_application}"确实可以应用聚类分析。')
    
    # 聚类与分类的区别
    st.subheader("聚类 vs. 分类：目标与方法的差异")
    
    # 关键区别
    st.markdown("""
    **关键区别：**
    
    * **分类 (Supervised Learning):** **有**预定义的类别标签，目标是学习一个模型，将新数据点分配到这些已知类别中。
    
    * **聚类 (Unsupervised Learning):** **没有**预定义的类别标签，目标是发现数据本身的内在结构，自动将数据分成簇。
    """)
    
    # 对比表格
    comparison = {
        "特征": [
            "学习类型", 
            "输入数据", 
            "目标", 
            "输出", 
            "评估"
        ],
        "分类 (Classification)": [
            "有监督学习 (Supervised)", 
            "带标签的数据 (X, y)", 
            "学习从 X到 y 的映射函数", 
            "新数据点的类别预测", 
            "准确率、精确率、召回率、F1 分数等"
        ],
        "聚类 (Clustering)": [
            "无监督学习 (Unsupervised)", 
            "不带标签的数据 (X)", 
            "发现数据 X 中的分组结构", 
            "数据点的簇分配", 
            "轮廓系数、DB 指数、纯度 (需标签) 等"
        ]
    }
    
    df_comparison = pd.DataFrame(comparison)
    st.table(df_comparison)
    
    # 讨论问题
    st.markdown("""
    **讨论：** 想象一个分析银行客户流失的场景。我们应该使用分类还是聚类？为什么？
    """)
    
    classification_reason = """
    如果目标是预测哪些客户**将要**流失，我们需要历史流失标签，即哪些客户已经流失，哪些仍然活跃。
    这是一个有监督学习问题，应该使用**分类**算法。
    """
    
    clustering_reason = """
    如果目标是了解**现有**客户可以分成哪些群体，以便针对性营销，我们不需要预先定义的标签。
    这是一个无监督学习问题，应该使用**聚类**算法。
    """
    
    # 创建可折叠区域
    with st.expander("需要提示？"):
        st.markdown("""
        考虑两种情况：
        1. 我们想知道哪些现有客户将来可能流失
        2. 我们想将客户分组，了解不同类型的客户行为模式
        """)
    
    # 让用户选择答案
    use_case = st.radio("在银行客户流失分析中，哪种情况下应该使用聚类？", 
                         ["预测哪些客户将要流失", "了解现有客户可以分成哪些群体"])
    
    if st.button("检查答案"):
        if use_case == "了解现有客户可以分成哪些群体":
            st.success("正确！" + clustering_reason)
        else:
            st.error("不完全正确。" + classification_reason)
            st.info("而" + clustering_reason)
    
    # 简单测验
    create_quiz(
        "以下哪项任务最适合使用聚类分析？",
        [
            "根据历史电子邮件将新邮件分类为垃圾邮件或非垃圾邮件",
            "根据购买历史预测客户是否会购买特定产品",
            "根据购物行为将客户分成不同群体以进行针对性营销",
            "根据社交媒体数据预测用户的年龄段"
        ],
        2,  # 正确答案的索引
        "聚类是一种无监督学习方法，适用于无需预先定义标签的分组任务。根据购物行为将客户分组是典型的聚类应用，因为我们不知道有哪些组，而是想从数据中发现自然分组。"
    ) 