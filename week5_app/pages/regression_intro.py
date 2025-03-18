import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from pathlib import Path
import sys

# 添加utils目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.fonts import configure_matplotlib_fonts

def show():
    # 配置matplotlib字体
    configure_matplotlib_fonts()
    
    st.markdown("## 回归问题概述")
    
    st.markdown("### 什么是回归？")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        **回归分析**是一种预测分析，研究自变量（特征）与因变量（目标）之间的关系，目标是**预测连续型数值**。
        
        回归是最基础也是最重要的机器学习算法之一，它能够帮助我们：
        * 理解变量之间的关系
        * 预测未来的数值
        * 发现影响目标的关键因素及其影响程度
        """)
    
    with col2:
        # 这里应该放一个回归的示意图
        # 由于没有直接的图片资源，我们可以创建一个简单的回归示意图
        fig, ax = plt.subplots(figsize=(5, 3))
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2 * x + 1 + np.random.normal(0, 2, 50)
        ax.scatter(x, y, alpha=0.7)
        ax.plot(x, 2 * x + 1, 'r', label='真实关系')
        ax.set_xlabel('特征 X')
        ax.set_ylabel('目标 y')
        ax.set_title('回归示例')
        ax.legend()
        st.pyplot(fig)
    
    st.markdown("### 回归 vs 分类")
    
    comparison_data = {
        "特性": ["目标变量", "输出空间", "算法示例", "评估指标", "应用场景"],
        "回归": ["连续值", "实数域", "线性回归、多项式回归、Ridge回归、Lasso回归、决策树回归、随机森林回归、XGBoost回归...", 
                "均方误差(MSE)、均方根误差(RMSE)、平均绝对误差(MAE)、R²", 
                "房价预测、销量预测、温度预测..."],
        "分类": ["离散类别", "有限集合", "逻辑回归、决策树、随机森林、SVM、KNN、朴素贝叶斯、XGBoost分类...", 
                "准确率、精确率、召回率、F1值、AUC-ROC", 
                "垃圾邮件检测、信用评估、疾病诊断..."]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df)
    
    st.markdown("### 回归的常见应用场景")
    
    scenarios = {
        "房价预测": "基于房屋特征（如面积、位置、房间数量）预测房屋价格",
        "销量预测": "基于历史数据、促销活动、季节等因素预测未来销量",
        "股票价格预测": "基于历史价格和交易量等数据预测股票价格走势",
        "能源消耗估计": "预测建筑或企业的能源消耗量",
        "医疗健康": "预测患者的住院时间、药物反应程度等",
        "广告效果预测": "预测不同广告渠道和投入的回报率",
        "人力资源": "预测员工绩效或薪资水平"
    }
    
    scenario_df = pd.DataFrame({
        "应用场景": list(scenarios.keys()),
        "描述": list(scenarios.values())
    })
    
    st.table(scenario_df)
    
    st.markdown("### 回归问题的基本流程")
    
    st.markdown("""
    回归分析的基本流程包括以下步骤：
    
    1. **数据收集与预处理**
       * 收集相关数据
       * 处理缺失值和异常值
       * 特征工程（转换、缩放、编码）
    
    2. **探索性数据分析**
       * 查看特征分布
       * 分析特征间相关性
       * 可视化数据关系
    
    3. **模型选择与训练**
       * 选择合适的回归算法
       * 划分训练集和测试集
       * 训练模型
    
    4. **模型评估与调优**
       * 使用评估指标评估模型性能
       * 交叉验证
       * 超参数调优
    
    5. **模型解释与部署**
       * 解释特征重要性
       * 分析系数含义
       * 模型部署与监控
    """)
    
    st.markdown("### 回归模型选择指南")
    
    selection_data = {
        "情境": ["数据线性相关", "存在多重共线性", "数据非线性关系", "数据维度高、特征多", "需要特征选择", "需要高精度", "可解释性要求高", "计算资源有限"],
        "建议模型": ["线性回归", "Ridge回归", "多项式回归、决策树", "Lasso回归、ElasticNet", "Lasso回归", "集成方法（随机森林、XGBoost）", "线性回归、决策树", "线性模型"]
    }
    
    selection_df = pd.DataFrame(selection_data)
    st.table(selection_df)
    
    # 添加一个互动环节
    st.markdown("### 小测验")
    st.info("请选择以下哪种问题适合使用回归分析：")
    
    q1 = st.radio(
        "哪些是回归问题？（可多选）",
        ["预测明天的股票价格", "判断邮件是否为垃圾邮件", "预测用户购买某商品的可能性", "预测房屋的销售价格", "对图像进行分类"],
        index=0
    )
    
    if q1 == "预测明天的股票价格" or q1 == "预测房屋的销售价格":
        st.success("正确！预测连续值（如股票价格、房价）是典型的回归问题。")
    elif q1 == "预测用户购买某商品的可能性":
        st.info("部分正确。虽然输出是概率（0-1之间的连续值），但这通常被视为分类问题，因为最终目标是分类（购买/不购买）。")
    else:
        st.error("不正确。邮件分类和图像分类是典型的分类问题，而非回归问题。")

if __name__ == "__main__":
    show() 