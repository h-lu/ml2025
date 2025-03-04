import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.datasets import make_classification, make_circles, make_moons

# 设置页面标题和配置
st.set_page_config(
    page_title="机器学习分类算法 - 逻辑回归与SVM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "机器学习分类算法 - 逻辑回归与SVM 教学应用"
    }
)

# 导入各个功能模块
from theory import show_theory
from visualization import show_visualization
from basic_exercises import show_basic_exercises
from advanced_exercises import show_advanced_exercises
from ml_fundamentals import show_ml_fundamentals
from exercises import show_quizzes, show_learning_path, show_concept_explorer
from utils.fonts import configure_matplotlib_fonts, get_svg_style
from utils.styles import apply_modern_style, create_card

# 配置字体
configure_matplotlib_fonts()

# 应用现代苹果风格样式
apply_modern_style()

# 侧边栏导航
with st.sidebar:
    # 使用更适合机器学习课程的图标
    st.markdown("""
    <div style="text-align: center; margin-bottom: 10px;">
        <svg width="50" height="50" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="12" r="10" stroke="#4285F4" stroke-width="2"/>
            <circle cx="12" cy="12" r="6" stroke="#EA4335" stroke-width="2"/>
            <circle cx="12" cy="12" r="2" fill="#FBBC05"/>
            <path d="M12 4V8" stroke="#34A853" stroke-width="2"/>
            <path d="M12 16V20" stroke="#34A853" stroke-width="2"/>
            <path d="M4 12H8" stroke="#34A853" stroke-width="2"/>
            <path d="M16 12H20" stroke="#34A853" stroke-width="2"/>
        </svg>
    </div>
    """, unsafe_allow_html=True)
    st.title("机器学习课程")
    st.caption("第三周 · 分类算法学习")
    
    st.markdown("---")
    
    # 创建三个主要标签页
    st.subheader("导航菜单")
    
    navigation_options = {
        "🔍 机器学习基础": ["机器学习基础"],
        "📚 分类算法": ["理论介绍", "算法可视化", "基础练习", "综合练习"],
        "🔧 学习工具": ["概念探索", "学习路径", "知识测验"]
    }
    
    # 初始化页面选择
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "机器学习基础"
    
    # 创建可折叠的导航菜单
    for category, options in navigation_options.items():
        with st.expander(category, expanded=category=="🔍 机器学习基础"):
            for option in options:
                if st.button(option, key=f"btn_{option}", use_container_width=True):
                    st.session_state.current_page = option
    
    st.markdown("---")
    
    # 更新版本和作者信息，更适合教学风格
    st.markdown("""
    <div style='text-align: center; padding: 12px; border-radius: 5px; background-color: #f0f2f6; font-family: sans-serif;'>
        <div style='font-size: 0.9em; color: #555555;'>版本 v1.0.0</div>
        <div style='font-size: 0.8em; margin-top: 8px; color: #555555;'>© 2024 机器学习课程</div>
    </div>
    """, unsafe_allow_html=True)

# 主内容区域
st.title("分类算法学习 - 逻辑回归与支持向量机")

# 修改当前选择的页面指示器，使用更适合教学的设计
st.markdown(f"""
<div style='margin-bottom: 20px; padding: 10px 15px; background-color: #f0f2f6; border-radius: 5px; border-left: 5px solid #4285F4; font-family: sans-serif;'>
    <span style='font-weight: 500; color: #333333;'>当前学习内容:</span> <span style='color: #4285F4; font-weight: 500;'>{st.session_state.current_page}</span>
</div>
""", unsafe_allow_html=True)

# 根据选择显示不同的页面
current_page = st.session_state.current_page

if current_page == "理论介绍":
    show_theory()
elif current_page == "算法可视化":
    show_visualization()
elif current_page == "基础练习":
    show_basic_exercises()
elif current_page == "综合练习":
    show_advanced_exercises()
elif current_page == "机器学习基础":
    show_ml_fundamentals()
elif current_page == "概念探索":
    show_concept_explorer()
elif current_page == "学习路径":
    show_learning_path()
elif current_page == "知识测验":
    show_quizzes()

# 页脚，使用更适合教学的设计
st.markdown("---")
footer_html = create_card(
    "机器学习课程",
    """
    <div style="display: flex; justify-content: space-between; align-items: center; font-family: sans-serif;">
        <div style="color: #555555;">基于Streamlit构建的交互式学习平台</div>
        <div style="color: #555555;">© 2024 版权所有</div>
    </div>
    """
)
st.markdown(footer_html, unsafe_allow_html=True) 