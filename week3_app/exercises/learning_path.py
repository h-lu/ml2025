import streamlit as st
import os
from utils.svg_generator import render_svg
from utils.assessment import LearningAssessment
from utils.matplotlib_charts import create_learning_path_chart

def show_learning_path():
    """显示分类算法的学习路径"""
    st.header("分类算法学习路径")
    
    # 显示学习路径图 - 使用matplotlib代替SVG
    st.markdown("### 学习路径概览")
    learning_path_fig = create_learning_path_chart()
    st.pyplot(learning_path_fig)
    
    # 显示学习进度
    assessment = LearningAssessment()
    with st.expander("查看个人学习进度", expanded=False):
        assessment.show_progress_dashboard()
    
    # 学习阶段
    st.markdown("## 学习阶段与资源")
    
    # 第一阶段：基础理论
    st.markdown("### 1. 基础理论")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        #### 学习目标
        - 理解逻辑回归的数学原理与概率解释
        - 掌握SVM的最大间隔分类器概念
        - 了解两种算法的优缺点与适用场景
        
        #### 关键概念
        - Sigmoid函数和对数几率
        - 最大似然估计
        - 核函数与支持向量
        - 软间隔与正则化
        """)
    
    with col2:
        st.markdown("#### 推荐资源")
        st.markdown("""
        - 应用内"理论介绍"部分
        - 机器学习(周志华)第3章
        - 统计学习方法(李航)第5-6章
        - [Andrew Ng课程视频](https://www.coursera.org/learn/machine-learning)
        """)
    
    theory_progress = st.slider("理论学习完成度", 0, 100, 0, key="theory_slider") / 100
    if st.button("更新理论学习进度"):
        assessment.update_theory_progress("logistic_regression", theory_progress)
        assessment.update_theory_progress("svm", theory_progress)
        st.success(f"更新成功！当前理论学习进度：{theory_progress*100}%")
    
    # 第二阶段：可视化理解
    st.markdown("---")
    st.markdown("### 2. 可视化理解")
    
    st.markdown("""
    #### 学习目标
    - 通过可视化直观理解算法的决策边界
    - 观察不同参数对模型性能的影响
    - 理解过拟合与欠拟合现象
    
    #### 关键活动
    - 使用应用内"算法可视化"模块探索不同数据集上的模型表现
    - 尝试调整参数观察决策边界的变化
    - 比较线性与非线性核在不同数据集上的效果
    """)
    
    st.markdown("#### 实践建议")
    st.markdown("""
    1. 首先使用线性可分数据集，观察最基本的决策边界形态
    2. 然后尝试非线性数据集(如同心圆)，比较线性与非线性模型的表现差异
    3. 调整C参数，观察正则化对模型复杂度的影响
    4. 对于SVM，尝试不同的核函数，特别关注RBF核的gamma参数影响
    """)
    
    # 第三阶段：基础练习
    st.markdown("---")
    st.markdown("### 3. 基础练习")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        #### 学习目标
        - 掌握使用scikit-learn实现分类算法的基本流程
        - 了解基本的模型评估方法
        - 实践参数调整及其影响
        
        #### 关键步骤
        - 数据预处理和特征工程
        - 模型训练与参数选择
        - 使用各种指标评估模型
        """)
    
    with col2:
        st.markdown("#### 练习推荐")
        st.markdown("""
        完成应用内的基础练习：
        - 练习1：乳腺癌数据分类
        - 练习2：鸢尾花分类
        
        进阶挑战：
        - 尝试更多的特征工程方法
        - 实现交叉验证评估模型
        """)
    
    basic_ex_progress = st.slider("基础练习完成度", 0, 100, 0, key="basic_ex_slider") / 100
    if st.button("更新基础练习进度"):
        assessment.update_exercise_progress("basic_1", basic_ex_progress)
        assessment.update_exercise_progress("basic_2", basic_ex_progress)
        st.success(f"更新成功！当前基础练习进度：{basic_ex_progress*100}%")
    
    # 第四阶段：深入理解偏差-方差
    st.markdown("---")
    st.markdown("### 4. 深入理解偏差-方差")
    
    st.markdown("""
    #### 学习目标
    - 理解偏差-方差权衡的概念
    - 掌握如何诊断模型的过拟合与欠拟合问题
    - 学习使用学习曲线分析模型性能
    
    #### 关键概念
    - 高偏差(欠拟合)与高方差(过拟合)
    - 模型复杂度与泛化能力
    - 学习曲线与验证曲线
    
    #### 实践建议
    1. 访问应用中的"机器学习基础"部分，学习偏差-方差相关内容
    2. 使用不同复杂度的模型解决同一问题，比较结果
    3. 绘制并分析学习曲线，判断是否存在高偏差或高方差问题
    """)
    
    # 第五阶段：过拟合处理
    st.markdown("---")
    st.markdown("### 5. 过拟合处理")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        #### 学习目标
        - 掌握正则化方法防止过拟合
        - 理解特征选择对模型表现的影响
        - 学习交叉验证进行模型选择
        
        #### 关键技术
        - L1与L2正则化(LASSO与Ridge)
        - 特征选择与降维
        - 网格搜索与随机搜索
        """)
    
    with col2:
        st.markdown("#### 实践建议")
        st.markdown("""
        1. 在逻辑回归中尝试不同的正则化强度(C值)
        2. 对比L1和L2正则化的效果差异
        3. 结合PCA降维与模型训练
        4. 使用GridSearchCV寻找最优参数
        """)
    
    # 第六阶段：综合实践
    st.markdown("---")
    st.markdown("### 6. 综合实践")
    
    st.markdown("""
    #### 学习目标
    - 应用所学知识解决真实世界的分类问题
    - 掌握完整的机器学习项目流程
    - 提升模型调优与结果分析能力
    
    #### 关键活动
    - 完成应用内的"综合练习：收入预测"
    - 实践完整的数据科学流程：从数据清洗到模型部署
    - 尝试组合多种模型技术提升性能
    """)
    
    advanced_ex_progress = st.slider("综合练习完成度", 0, 100, 0, key="advanced_ex_slider") / 100
    if st.button("更新综合练习进度"):
        assessment.update_exercise_progress("advanced_1", advanced_ex_progress)
        st.success(f"更新成功！当前综合练习进度：{advanced_ex_progress*100}%")
    
    # 学习资源
    st.markdown("---")
    st.markdown("## 推荐学习资源")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 书籍")
        st.markdown("""
        - 《机器学习》周志华
        - 《统计学习方法》李航
        - 《Python机器学习》Sebastian Raschka
        - 《机器学习实战》Peter Harrington
        """)
        
        st.markdown("### 课程")
        st.markdown("""
        - Coursera: Machine Learning by Andrew Ng
        - 吴恩达深度学习专项课程
        - CS229: Machine Learning (Stanford)
        - 林轩田：机器学习基石与技法
        """)
    
    with col2:
        st.markdown("### 在线资源")
        st.markdown("""
        - [scikit-learn官方文档](https://scikit-learn.org/stable/modules/classes.html)
        - [Kaggle上的分类问题实例](https://www.kaggle.com/competitions)
        - [机器学习速查表](https://ml-cheatsheet.readthedocs.io/en/latest/)
        - [Distill.pub可视化解释](https://distill.pub/)
        """)
        
        st.markdown("### 工具")
        st.markdown("""
        - Jupyter Notebook/Lab
        - Google Colab
        - TensorBoard
        - MLflow
        """)
    
    # 自我测验
    st.markdown("---")
    st.markdown("## 自我测验")
    
    st.markdown("""
    完成测验可以帮助你检验学习成果，发现知识盲点。请访问"测验"部分完成以下测验：
    
    1. 逻辑回归基础测验
    2. SVM基础测验
    3. 模型比较测验
    4. 实践应用测验
    5. 编程实践测验
    
    每个测验包含5个问题，涵盖从基础理论到实际应用的各个方面。
    """)
    
    # 后续学习建议
    st.markdown("---")
    st.markdown("## 后续学习建议")
    
    st.markdown("""
    掌握逻辑回归和SVM后，你可以:
    
    1. 学习更多分类算法：
       - 决策树与随机森林
       - 朴素贝叶斯
       - KNN分类器
       
    2. 深入学习特征工程技巧：
       - 特征选择方法
       - 特征编码
       - 特征交互
       
    3. 探索深度学习领域：
       - 多层感知机与神经网络基础
       - CNN, RNN等高级网络
       - 迁移学习
    """)
    
    # 学习小贴士
    st.markdown("---")
    st.info("""
    **学习小贴士**：最有效的学习方式是理论与实践相结合。请尝试将所学知识应用到实际数据集上，
    并参与Kaggle等平台的竞赛来巩固技能。记住，在机器学习中，实践经验与理论同等重要。
    """) 