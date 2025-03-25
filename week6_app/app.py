import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import platform
import os
import matplotlib.font_manager as fm
import base64

# 设置页面配置 - 必须是第一个st命令
st.set_page_config(
    page_title="集成学习与回归模型优化课件",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 导入各页面模块
from pages import theory, code_demo, parameter_tuning, exercises, project
from utils.plot_utils import configure_matplotlib_fonts, create_ensemble_learning_svg

# 设置全局字体和图形设置
def setup_application():
    # 配置matplotlib以支持中文
    configure_matplotlib_fonts()
    
    # 检测系统中文字体支持
    system = platform.system()
    found_chinese_font = False
    
    # 获取系统中所有可用的字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 检查是否有中文字体
    chinese_fonts = ['PingFang HK', 'PingFang SC', 'Heiti SC', 'STHeiti', 'Microsoft YaHei', 'SimHei']
    
    for font in chinese_fonts:
        if font in available_fonts:
            found_chinese_font = True
            st.sidebar.success(f"找到中文字体: {font}")
            break
    
    if not found_chinese_font:
        st.sidebar.warning("未找到有效的中文字体，图表中可能无法正确显示中文。将使用HTML元素代替。")
    
    # 设置Streamlit页面中文字体
    st.markdown("""
    <style>
        body {
            font-family: 'PingFang HK', 'PingFang SC', 'Microsoft YaHei', sans-serif;
        }
    </style>
    """, unsafe_allow_html=True)

# 初始化应用程序
setup_application()

# 配置matplotlib以支持中文
configure_matplotlib_fonts()

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .highlight {
        color: #1565C0;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# 侧边栏导航
st.sidebar.title("导航菜单")
pages = {
    "主页": "home",
    "理论学习": "theory",
    "代码演示": "code_demo",
    "参数调优实验": "parameter_tuning",
    "练习题": "exercises",
    "项目指导": "project"
}
selection = st.sidebar.radio("选择页面", list(pages.keys()))

# 侧边栏额外信息
st.sidebar.markdown("---")
st.sidebar.info("本课件基于第六周教学内容：集成学习与回归模型优化")
st.sidebar.markdown("作者：机器学习课程组")

# 页面路由
if pages[selection] == "home":
    # 主页内容
    st.markdown('<h1 class="main-header">第六周：集成学习与回归模型优化</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">本周学习目标</h2>', unsafe_allow_html=True)
        st.markdown("""
        * 理解集成学习的基本原理和分类
        * 掌握梯度提升决策树(GBDT)的工作原理
        * 深入学习XGBoost算法的特点和优势
        * 掌握XGBoost模型的参数调优方法
        * 能够使用xgboost库实现和评估回归模型
        * 学习回归模型评估指标的选择和应用场景
        * 将XGBoost应用于实际的房价预测问题
        """)
        
        st.markdown('<h2 class="sub-header">课件内容概览</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <p><span class="highlight">理论学习</span>：集成学习原理、GBDT基础、XGBoost算法详解</p>
        <p><span class="highlight">代码演示</span>：XGBoost实现与基础应用</p>
        <p><span class="highlight">参数调优实验</span>：交互式调整XGBoost参数并观察效果</p>
        <p><span class="highlight">练习题</span>：基础练习与扩展练习</p>
        <p><span class="highlight">项目指导</span>：房价预测模型优化项目指南</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # 使用SVG格式的集成学习示意图
        st.markdown("### 集成学习示意图")
        
        # 使用base64编码显示SVG图像
        try:
            # 读取SVG文件
            svg_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                        'utils', 'ensemble_learning.svg')
            with open(svg_file_path, 'r', encoding='utf-8') as f:
                svg_content = f.read()
            
            # 使用data URI方式显示SVG
            b64 = base64.b64encode(svg_content.encode('utf-8')).decode('utf-8')
            
            # 创建HTML图像标签
            html = f'''
            <div style="display:flex; justify-content:center;">
                <img src="data:image/svg+xml;base64,{b64}" width="400" alt="集成学习流程图">
            </div>
            '''
            st.markdown(html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"无法加载SVG图片: {e}")
            # 显示备用内容
            st.markdown("""
            <div style="width:400px; height:320px; border:1px solid #ddd; border-radius:5px; display:flex; 
                 flex-direction:column; justify-content:center; align-items:center; background-color:#f8f9fa; 
                 margin:0 auto; font-family:'PingFang HK', 'Microsoft YaHei', sans-serif;">
                <div style="font-size:18px; color:#1E88E5; margin-bottom:20px;">集成学习流程图</div>
                <div style="color:#666; text-align:center; padding:0 20px;">
                    集成多个模型的预测结果,<br>提高整体性能和泛化能力
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # 添加一个简单的集成学习优势说明
        st.markdown("""
        <div style="background-color:#F1F8E9; padding:10px; border-radius:5px; margin-top:20px;">
        <p><b>集成学习优势：</b></p>
        <ul>
          <li>提高模型精度</li>
          <li>减少过拟合风险</li>
          <li>增强模型稳定性</li>
          <li>处理复杂数据关系</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

elif pages[selection] == "theory":
    theory.show()

elif pages[selection] == "code_demo":
    code_demo.show()

elif pages[selection] == "parameter_tuning":
    parameter_tuning.show()

elif pages[selection] == "exercises":
    exercises.show()

elif pages[selection] == "project":
    project.show() 