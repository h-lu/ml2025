import streamlit as st
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import platform

# 添加模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 配置matplotlib支持中文字体
def configure_matplotlib_fonts():
    """
    根据操作系统配置matplotlib的中文字体支持
    """
    system = platform.system()
    
    # 设置全局字体
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    
    # 修复负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    
    # 支持SVG输出
    plt.rcParams['svg.fonttype'] = 'none'
    
    # 获取系统可用的字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 根据不同系统设置中文字体
    if system == 'Windows':
        # Windows系统
        font_candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi', 'Arial Unicode MS']
    elif system == 'Darwin':  # macOS
        # macOS系统常见中文字体
        font_candidates = ['PingFang HK', 'PingFang SC', 'PingFang TC', 'Heiti SC', 'STHeiti', 'Hiragino Sans GB', 'Apple LiGothic']
    else:  # Linux和其他系统
        # Linux系统
        font_candidates = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback', 'Noto Sans CJK SC', 'DejaVu Sans']
    
    # 选择第一个可用的字体
    found_font = False
    for font in font_candidates:
        if font in available_fonts:
            print(f"使用中文字体: {font}")
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams.get('font.sans-serif', [])
            found_font = True
            break
    
    if not found_font:
        print("未找到合适的中文字体，将使用默认字体")
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica'] + plt.rcParams.get('font.sans-serif', [])

# 配置中文字体
configure_matplotlib_fonts()

# 设置页面配置
st.set_page_config(
    page_title="聚类分析基础：K-means 与层次聚类",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置应用标题
st.title("深入探索聚类分析：K-means 与层次聚类")
st.markdown("---")

# 侧边栏导航
st.sidebar.title("导航")
page = st.sidebar.radio(
    "选择章节:",
    ["课程目标", "聚类分析导论", "K-means 聚类", 
     "层次聚类", "聚类评估与算法选择", "实践环节", "总结与反思"]
)

# 导入各模块
from pages.intro import show_intro, show_clustering_intro
from pages.kmeans import show_kmeans
from pages.hierarchical import show_hierarchical
from pages.evaluation import show_evaluation
from pages.practice import show_practice

# 在主界面显示选定的页面内容
if page == "课程目标":
    show_intro()
elif page == "聚类分析导论":
    show_clustering_intro()
elif page == "K-means 聚类":
    show_kmeans()
elif page == "层次聚类":
    show_hierarchical()
elif page == "聚类评估与算法选择":
    show_evaluation()
elif page == "实践环节":
    show_practice()
elif page == "总结与反思":
    st.header("总结、反思与展望")
    
    st.subheader("核心回顾")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        * 聚类的本质是发现数据分组。
        * K-means：简单高效，但有局限性（K 值、初始化、形状假设）。
        * 层次聚类：提供结构，无需预设 K，但计算成本高。
        """)
    with col2:
        st.markdown("""
        * 评估是关键：内部、外部、可视化结合。
        * 算法选择需权衡各种因素。
        """)
    
    st.subheader("批判性思考")
    st.markdown("""
    * 聚类结果总是"有意义"的吗？
      * 可能发现的是算法偏好而非真实结构
    * 不同算法在同一数据上给出不同结果，哪个是对的？
      * 没有绝对对错，看哪个更符合分析目标和数据特性
    """)
    
    st.subheader("课后探索")
    st.markdown("""
    * 尝试使用 `scikit-learn` 对 Iris 数据集进行 K-means 和层次聚类，比较结果。
    * 研究 K-means++ 初始化方法。
    * 了解 DBSCAN 等基于密度的聚类算法，它们能处理任意形状的簇。
    """)
    
    # 自评表
    st.subheader("学习目标自评")
    objectives = [
        "深入理解聚类分析的核心概念、目标和应用场景",
        "熟练掌握 K-means 聚类算法的原理和步骤",
        "清晰理解凝聚型层次聚类的原理和不同 Linkage 方法",
        "掌握常用的聚类评估指标及其应用场景",
        "能够根据数据特点和分析目标选择合适的聚类算法",
        "初步了解在 Python 中实现基本聚类算法的方法"
    ]
    
    for objective in objectives:
        level = st.slider(
            f"{objective}", 
            min_value=1, 
            max_value=5, 
            value=3,
            help="1=完全不理解, 5=完全掌握"
        )

# 添加页脚
st.markdown("---")
st.markdown("© 2024 机器学习课程 | 聚类分析基础") 