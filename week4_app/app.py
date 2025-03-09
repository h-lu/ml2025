import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# 导入页面模块
from pages import decision_tree, ensemble, evaluation, basic_exercises, advanced_exercises

# 设置页面配置
st.set_page_config(
    page_title="第四周：决策树与随机森林",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #42A5F5;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .objective-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #66BB6A;
    }
</style>
""", unsafe_allow_html=True)

# 应用标题
st.markdown('<h1 class="main-header">第四周：分类算法基础（二）- 决策树与集成学习（随机森林）</h1>', unsafe_allow_html=True)

# 侧边栏导航
st.sidebar.title("导航")
page = st.sidebar.radio(
    "选择内容", 
    ["课程介绍", "决策树算法", "集成学习与随机森林", "模型评估与选择", "基础练习", "扩展练习"]
)

# 学习目标
if page == "课程介绍":
    st.markdown('<h2 class="sub-header">本周学习目标</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="objective-box">', unsafe_allow_html=True)
    st.markdown("""
    * 掌握决策树算法的原理、信息增益/基尼系数的计算方法和 Scikit-learn 实现。
    * 理解决策树的优缺点，并能够进行可视化展示。
    * 掌握集成学习 Bagging 方法和随机森林算法的原理和 Scikit-learn 实现。
    * 理解随机森林的优缺点和特征重要性的概念。
    * 回顾分类模型评估指标，并深入理解交叉验证和网格搜索的模型选择与调优方法。
    * 能够根据不同的应用场景选择合适的分类模型评估指标。
    * 使用 Scikit-learn 构建、评估和调优决策树和随机森林分类模型。
    * 比较不同分类模型在电商用户行为数据集上的性能。
    * 使用随机森林算法优化电商用户行为分类模型。
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-header">内容概要</h2>', unsafe_allow_html=True)
    st.markdown("""
    本周我们将学习两种重要的分类算法：决策树和随机森林。首先介绍决策树的基本原理、特征选择方法和Scikit-learn实现。然后讲解集成学习中的Bagging方法和随机森林算法。最后回顾和深入讨论模型评估指标以及模型选择与调优的方法。
    
    课程包括理论讲解和实践操作，将通过交互式示例帮助大家理解算法原理，并通过实际案例展示如何应用这些算法解决实际问题。
    """)
    
    st.markdown('<h2 class="sub-header">参考资料</h2>', unsafe_allow_html=True)
    st.markdown("""
    * Scikit-learn 官方文档 - 决策树: [链接](https://scikit-learn.org/stable/modules/tree.html)
    * Scikit-learn 官方文档 - 随机森林: [链接](https://scikit-learn.org/stable/modules/ensemble.html#forests)
    * Scikit-learn 官方文档 - 交叉验证: [链接](https://scikit-learn.org/stable/modules/cross_validation.html)
    * Scikit-learn 官方文档 - 网格搜索: [链接](https://scikit-learn.org/stable/modules/grid_search.html)
    * 《Python Data Science Handbook》: [链接](https://jakevdp.github.io/PythonDataScienceHandbook/) (Chapter 5 - Machine Learning)
    * 《Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow》: [链接](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125973/) (Chapter 6 - Decision Trees, Chapter 7 - Ensemble Learning and Random Forests)
    """)

# 加载选定页面
elif page == "决策树算法":
    decision_tree.show()
elif page == "集成学习与随机森林":
    ensemble.show()
elif page == "模型评估与选择":
    evaluation.show()
elif page == "基础练习":
    basic_exercises.show()
elif page == "扩展练习":
    advanced_exercises.show() 