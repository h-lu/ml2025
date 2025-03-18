import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

from utils.data_loader import load_dataset
from utils.visualization import (
    plot_decision_tree, 
    plot_feature_importance, 
    plot_confusion_matrix, 
    plot_roc_curve
)

def show():
    """显示决策树内容"""
    st.title("决策树算法")
    
    st.markdown("""
    ### 1. 决策树简介
    
    决策树是一种基本的分类与回归方法，它呈现的是一种树形结构。决策树的每个内部节点表示一个属性上的测试，每个分支代表一个测试输出，每个叶节点代表一种类别。
    
    决策树学习的目标就是为了产生一棵泛化能力强，即处理未见示例能力强的决策树。
    """)
    
    # 获取当前文件所在的目录路径
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取week4_app目录路径（与run.py同级）
    app_dir = os.path.dirname(current_file_dir)
    
    # 构建SVG文件的绝对路径（只查找week4_app/img/目录）
    svg_path = os.path.join(app_dir, "img", "decision_tree_concept.svg")
    
    # 概念图
    st.markdown("#### 决策树概念图")
    try:
        # 检查文件是否存在
        if os.path.exists(svg_path):
            st.image(svg_path, caption="决策树概念示意图", use_column_width=True)
        else:
            st.error(f"找不到决策树概念图文件，路径：{svg_path}")
            # 显示纯文本描述作为备选
            st.markdown("""
            **决策树简单示例**:
            ```
            是否有收入 > 5000?
            ├── 是: 是否有房产?
            │   ├── 是: 批准贷款 ✓
            │   └── 否: 是否有担保人?
            │       ├── 是: 批准贷款 ✓
            │       └── 否: 拒绝贷款 ✗
            └── 否: 拒绝贷款 ✗
            ```
            """)
    except Exception as e:
        st.error(f"加载决策树概念图时出错: {e}")
    
    # 创建选项卡
    tabs = st.tabs(["算法原理", "特征选择", "Scikit-learn实现", "优缺点", "交互式演示"])
    
    # 算法原理选项卡
    with tabs[0]:
        st.markdown("""
        ## 决策树的原理
        
        决策树是一种树形结构的分类模型，每个内部节点表示一个特征的测试，每个分支代表一个测试输出，每个叶节点代表一个类别。
        
        ### 树形结构
        
        决策树的结构类似于流程图，从根节点开始，通过对特征的判断，沿着分支向下，最终到达叶节点，得到分类结果。
        """)
        
        # 使用自定义SVG图代替外部图像链接
        try:
            if os.path.exists(svg_path):
                st.image(svg_path, caption="决策树概念示意图", use_column_width=True)
            else:
                st.error(f"找不到决策树概念图文件，路径：{svg_path}")
        except Exception as e:
            st.error(f"加载决策树概念图时出错: {e}")
        
        st.markdown("""
        ### 决策过程
        
        1. 从根节点开始，根据样本在每个节点上的特征取值，递归地将样本分到不同的分支
        2. 直到到达叶节点，叶节点对应的类别即为预测结果
        
        ### 非线性模型
        
        决策树可以处理非线性数据，能够进行复杂的分类。它通过将特征空间划分为多个区域，每个区域对应一个类别，从而实现分类。
        """)
    
    # 特征选择选项卡
    with tabs[1]:
        st.markdown("""
        ## 特征选择（分裂准则）
        
        决策树算法的关键在于如何选择最优的特征进行节点分裂。**"最优"特征是指能够最大程度提高数据纯度的特征。** 常用的分裂准则包括：
        """)
        
        # 创建两列布局
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 信息增益 (Information Gain)
            
            基于信息熵 (Entropy) 的分裂准则。**信息熵** 衡量了数据集的混乱程度，熵越高，数据越"乱"，纯度越低。**信息增益** 表示使用某个特征进行分裂后，数据集信息熵减少的程度。
            
            #### 信息熵 (Entropy) 公式:
            
            $Entropy(S) = - \sum_{i=1}^{c} p_i \log_2(p_i)$
            
            其中 $S$ 是数据集， $p_i$ 是类别 $i$ 在数据集 $S$ 中所占的比例，$c$ 是类别数量。**熵值越大，数据集纯度越低。**
            
            #### 信息增益 (Information Gain) 公式:
            
            $Gain(S, A) = Entropy(S) - \\sum_{v \\in Values(A)} \\frac{|S_v|}{|S|} Entropy(S_v)$
            
            其中 $A$ 是特征， $Values(A)$ 是特征 $A$ 的取值集合， $S_v$ 是特征 $A$ 取值为 $v$ 的子数据集。
            
            > **信息增益直观理解:** 信息增益就像是"提纯"数据的能力。选择信息增益大的特征，就像是用更有效的"筛子"来筛选数据，使得相同类别的样本更集中在一起，不同类别的样本更容易区分开。
            """)
        
        with col2:
            st.markdown("""
            ### 基尼系数 (Gini Impurity)
            
            基于基尼指数 (Gini Index) 的分裂准则。**基尼指数** 衡量了数据集的不纯度，基尼指数越小，数据纯度越高。
            
            #### 基尼指数 (Gini Index) 公式:
            
            $Gini(S) = 1 - \sum_{i=1}^{c} p_i^2$
            
            其中 $S$ 是数据集， $p_i$ 是类别 $i$ 在数据集 $S$ 中所占的比例，$c$ 是类别数量。**基尼指数越大，数据集纯度越低。**
            
            #### 基尼系数增益:
            
            类似于信息增益，基尼系数增益表示使用某个特征分裂后，基尼系数下降的程度。决策树算法会选择基尼系数增益最大的特征进行分裂。
            
            > **基尼系数直观理解:** 基尼系数可以理解为衡量数据"杂乱"程度的指标。选择基尼系数下降最快的特征，就像是用最有效的"梳子"来梳理数据，使得数据更有条理，相同类别的样本更"抱团"，不同类别的样本更"分离"。
            """)
        
        st.image("https://miro.medium.com/max/1400/1*fU_ixlNBGJ-YBVRmOJ_6XQ.png", caption="信息熵与基尼系数对比")
    
    # Scikit-learn实现选项卡
    with tabs[2]:
        st.markdown("""
        ## Scikit-learn 实现决策树
        
        使用 `sklearn.tree.DecisionTreeClassifier` 类可以轻松实现决策树分类器。
        
        ### 主要参数
        
        - `criterion`: 分裂准则，可选 'gini' 或 'entropy'，默认为 'gini'
        - `max_depth`: 树的最大深度，默认为 None（无限制）
        - `min_samples_split`: 分裂内部节点所需的最小样本数，默认为 2
        - `min_samples_leaf`: 叶节点所需的最小样本数，默认为 1
        - `max_features`: 寻找最佳分裂时考虑的特征数量，默认为 None（所有特征）
        
        ### 示例代码
        """)
        
        st.code("""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import export_graphviz
import graphviz

# 加载电商用户行为数据集 (假设已预处理完成)
data = pd.read_csv('ecommerce_user_behavior_preprocessed.csv')

# 假设 'label' 列为分类目标变量，其他列为特征变量
X = data.drop('label', axis=1)
y = data['label']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建 DecisionTreeClassifier 模型
dt_model = DecisionTreeClassifier(criterion='gini', max_depth=5)

# 训练模型
dt_model.fit(X_train, y_train)

# 预测测试集
y_pred = dt_model.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"决策树模型准确率: {accuracy:.4f}")
print("\\n分类报告:\\n", classification_report(y_test, y_pred))

# 可视化决策树
dot_data = export_graphviz(dt_model, out_file=None,
                           feature_names=X_train.columns,
                           class_names=[str(c) for c in dt_model.classes_],
                           filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree")  # 保存为 decision_tree.pdf
        """, language="python")
    
    # 优缺点选项卡
    with tabs[3]:
        st.markdown("""
        ## 决策树的优缺点
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 优点
            
            - **易于理解和解释**，树形结构可视化直观
            - **可以处理类别型和数值型数据**
            - **对数据预处理要求不高**，不需要特征缩放
            - **能够处理非线性关系**
            - **可以自动进行特征选择**，树的顶部节点通常是最重要的特征
            - **可以处理多分类问题**，不需要进行额外的编码
            """)
        
        with col2:
            st.markdown("""
            ### 缺点
            
            - **容易过拟合**，特别是当树的深度过大时
            - **对数据中的噪声和异常值比较敏感**
            - **决策树模型不稳定**，数据的小幅变动可能导致树结构发生很大变化
            - **可能产生偏向多值特征的偏差**
            - **难以学习某些关系**，如XOR关系
            - **可能创建有偏差的树**，如果某些类别在数据中占主导地位
            """)
    
    # 交互式演示选项卡
    with tabs[4]:
        st.markdown("## 交互式决策树演示")
        
        # 数据集选择
        dataset_name = st.selectbox(
            "选择数据集",
            ["iris", "wine", "breast_cancer", "ecommerce"],
            index=0
        )
        
        # 加载数据集
        X_train, X_test, y_train, y_test, feature_names, target_names = load_dataset(
            dataset_name, test_size=0.2, random_state=42
        )
        
        # 模型参数设置
        col1, col2 = st.columns(2)
        
        with col1:
            criterion = st.selectbox("分裂准则", ["gini", "entropy"], index=0)
            max_depth = st.slider("最大深度", min_value=1, max_value=20, value=3)
        
        with col2:
            min_samples_split = st.slider("最小分裂样本数", min_value=2, max_value=20, value=2)
            min_samples_leaf = st.slider("最小叶节点样本数", min_value=1, max_value=20, value=1)
        
        # 训练决策树模型
        dt_model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        dt_model.fit(X_train, y_train)
        
        # 模型评估
        y_pred = dt_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.markdown(f"### 模型准确率: {accuracy:.4f}")
        
        # 可视化决策树
        st.markdown("### 决策树可视化")
        plot_decision_tree(dt_model, feature_names, target_names, max_depth=3)
        
        # 特征重要性
        st.markdown("### 特征重要性")
        plot_feature_importance(dt_model, feature_names)
        
        # 混淆矩阵
        st.markdown("### 混淆矩阵")
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, target_names)
        
        # 如果是二分类问题，绘制ROC曲线
        if len(target_names) == 2:
            st.markdown("### ROC曲线")
            y_prob = dt_model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plot_roc_curve(fpr, tpr, roc_auc) 