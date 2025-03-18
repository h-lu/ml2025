import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

# 导入字体配置
from utils.fonts import configure_matplotlib_fonts

# 配置matplotlib字体
configure_matplotlib_fonts()

# 导入页面模块
from pages import regression_intro, linear_regression, polynomial_regression, evaluation, basic_exercises, advanced_exercises

# 设置页面配置
st.set_page_config(
    page_title="第五周：回归算法基础",
    page_icon="📈",
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
st.markdown('<h1 class="main-header">第五周：回归算法基础 - 线性回归与多项式回归</h1>', unsafe_allow_html=True)

# 侧边栏导航
st.sidebar.title("导航")
page = st.sidebar.radio(
    "选择内容", 
    ["课程介绍", "回归问题概述", "线性回归", "多项式回归", "回归模型评估", "基础练习", "扩展练习"]
)

# 学习目标
if page == "课程介绍":
    st.markdown('<h2 class="sub-header">本周学习目标</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="objective-box">', unsafe_allow_html=True)
    st.markdown("""
    * 掌握回归算法的基本原理和应用场景
    * 理解线性回归和多项式回归的区别与联系
    * 掌握回归模型的评估指标和方法
    * 学习正则化技术在回归中的应用
    * 能够使用scikit-learn实现和评估回归模型
    * 将学到的回归算法应用于实际的房价预测问题
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-header">内容概要</h2>', unsafe_allow_html=True)
    st.markdown("""
    本周我们将学习回归算法的基础知识。首先介绍回归问题的概念和应用场景，然后深入学习线性回归和多项式回归的原理和实现方法。接着讨论回归模型的评估指标和正则化技术，最后通过实际案例展示如何应用这些算法解决房价预测问题。
    
    课程包括理论讲解和实践操作，将通过交互式示例帮助大家理解算法原理，并通过实际案例展示如何应用这些算法解决实际问题。
    """)
    
    st.markdown('<h2 class="sub-header">参考资料</h2>', unsafe_allow_html=True)
    st.markdown("""
    * Scikit-learn 官方文档 - 线性回归: [链接](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)
    * Scikit-learn 官方文档 - 多项式回归: [链接](https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions)
    * Scikit-learn 官方文档 - 正则化: [链接](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification)
    * 《Python Data Science Handbook》: [链接](https://jakevdp.github.io/PythonDataScienceHandbook/) (Chapter 5 - Machine Learning)
    * 《Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow》: [链接](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) (Chapter 4 - Training Models)
    """)

# 加载选定页面
elif page == "回归问题概述":
    regression_intro.show()
elif page == "线性回归":
    linear_regression.show()
elif page == "多项式回归":
    polynomial_regression.show()
elif page == "回归模型评估":
    evaluation.show()
elif page == "基础练习":
    basic_exercises.show()
elif page == "扩展练习":
    advanced_exercises.show() 