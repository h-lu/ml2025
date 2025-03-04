"""
机器学习基础介绍模块
"""

import streamlit as st
import os
from utils.fonts import get_svg_style
from utils.svg_generator import create_ml_basics_concept_svg, create_ml_workflow_svg, render_svg

def show_ml_introduction():
    """显示机器学习基础介绍内容"""
    
    st.subheader("机器学习简介")
    
    st.markdown("""
    **机器学习**是人工智能的一个分支，它让计算机系统能够通过数据学习和改进，而无需显式编程。
    机器学习算法可以从数据中发现模式，并使用这些模式进行预测或决策。
    """)
    
    # 显示机器学习基本概念图
    st.markdown("### 机器学习的基本概念")
    
    # 使用SVG路径
    svg_path = os.path.join("img", "ml_basics_concept.svg")
    
    # 检查图片是否存在，不存在则创建
    if not os.path.exists(svg_path):
        # 确保img目录存在
        os.makedirs(os.path.dirname(svg_path), exist_ok=True)
        
        # 创建SVG内容并保存到文件
        svg_content = create_ml_basics_concept_svg()
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
    
    # 直接读取SVG内容并使用render_svg函数显示
    try:
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
        render_svg(svg_content)
    except Exception as e:
        st.error(f"显示SVG图片时出错: {str(e)}")
        st.warning("生成图片文件失败，请检查路径和权限")
    
    st.markdown("""
    ### 机器学习的主要类型
    
    1. **监督学习**：算法从带标签的训练数据中学习，预测新数据的标签
       - 分类：预测离散类别（如垃圾邮件检测）
       - 回归：预测连续值（如房价预测）
    
    2. **无监督学习**：算法从无标签数据中学习，寻找数据中的隐藏结构
       - 聚类：将相似的数据点分组（如客户细分）
       - 降维：减少数据的特征数量，保留重要信息
    
    3. **强化学习**：算法通过与环境交互学习行为，以最大化累积奖励
       - 应用：游戏AI、机器人控制、自动驾驶
    """)
    
    # 显示机器学习工作流程图
    st.markdown("### 机器学习的一般工作流程")
    
    # 使用SVG路径
    svg_path = os.path.join("img", "ml_workflow.svg")
    
    # 检查图片是否存在，不存在则创建
    if not os.path.exists(svg_path):
        # 确保img目录存在
        os.makedirs(os.path.dirname(svg_path), exist_ok=True)
        
        # 创建SVG内容并保存到文件
        svg_content = create_ml_workflow_svg()
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
    
    # 直接读取SVG内容并使用render_svg函数显示
    try:
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
        render_svg(svg_content)
    except Exception as e:
        st.error(f"显示SVG图片时出错: {str(e)}")
        st.warning("生成图片文件失败，请检查路径和权限")
    
    st.markdown("""
    ### 机器学习面临的常见挑战
    
    1. **数据质量问题**: 缺失值、噪声、异常值
    2. **特征选择与工程**: 提取有意义的特征
    3. **过拟合与欠拟合**: 平衡模型复杂度
    4. **计算资源限制**: 大规模数据集的处理
    5. **模型可解释性**: 理解模型决策过程
    
    思考：在你的学习或工作中，你可能遇到过哪些数据分析或预测问题，可以用机器学习解决？
    """)
    
    # 提示学生思考
    st.info("""
    **思考问题**：
    
    1. 你能想到日常生活中至少3个使用机器学习的例子吗？
    2. 在学习机器学习的过程中，你认为最具挑战性的部分是什么？
    3. "数据是机器学习的燃料"，你如何理解这句话？
    """) 