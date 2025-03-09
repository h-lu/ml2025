import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score

from utils.data_loader import load_dataset
from utils.visualization import (
    plot_feature_importance, 
    plot_confusion_matrix, 
    plot_roc_curve,
    plot_cross_validation
)

def show():
    """显示集成学习内容，包括随机森林等算法"""
    st.title("集成学习与随机森林")
    
    st.markdown("""
    ### 1. 集成学习原理
    
    集成学习（Ensemble Learning）是机器学习中的一种方法，它通过组合多个基本模型的预测来提高整体预测性能。
    集成学习的基本思想是"三个臭皮匠，胜过诸葛亮"。
    
    集成学习主要分为两类：
    - **Bagging**：使用有放回抽样的方式从训练集中独立生成多个子训练集，然后在各个子训练集上训练基学习器，最后对这些基学习器的预测结果进行投票或平均。
    - **Boosting**：串行训练一系列基学习器，每个基学习器都试图修正前面基学习器的错误。
    """)
    
    # 随机森林概念图
    st.markdown("#### 随机森林概念图")
    try:
        # 尝试从不同的路径加载SVG文件
        try:
            st.image("img/random_forest_concept.svg", caption="随机森林集成学习示意图", use_column_width=True)
        except:
            st.image("week4_app/img/random_forest_concept.svg", caption="随机森林集成学习示意图", use_column_width=True)
    except Exception as e:
        st.error(f"加载图片时出错: {e}")
        # 提供文本描述作为备选
        st.markdown("""
        **随机森林原理简述**:
        
        1. 从原始数据集中有放回抽样，构建多个子数据集
        2. 对每个子数据集训练一个决策树，但在每个节点分裂时只考虑特征的随机子集
        3. 每棵树独立进行预测，最终结果通过投票或平均得出
        """)
    
    # 创建选项卡
    tabs = st.tabs(["集成学习基础", "Bagging方法", "随机森林算法", "优缺点", "交互式演示"])
    
    # 集成学习基础选项卡
    with tabs[0]:
        st.markdown("""
        ## 集成学习基础
        
        集成学习 (Ensemble Learning) 是将多个"弱学习器" (weak learner, 例如简单的决策树) 集成起来，构建一个"强学习器" (strong learner)，提高模型的泛化能力和鲁棒性。
        
        ### 核心思想
        
        "众人拾柴火焰高"，集成学习的核心思想就是"集思广益"，通过多个模型的共同决策来提高整体性能。
        
        ### 主要方法
        
        集成学习主要有三种方法：
        
        1. **Bagging (Bootstrap Aggregating)**: 通过有放回抽样构建多个训练集，训练多个基学习器，然后对它们的预测结果进行投票或平均。代表算法：随机森林 (Random Forest)。
        
        2. **Boosting**: 通过迭代训练一系列的基学习器，每个新的基学习器都尝试纠正前一个学习器的错误。代表算法：AdaBoost, Gradient Boosting, XGBoost。
        
        3. **Stacking**: 训练多个不同的基学习器，然后再训练一个元学习器，将基学习器的预测结果作为输入，输出最终的预测结果。
        
        本周我们主要学习Bagging方法和随机森林算法。
        """)
        
        # 使用自定义SVG图
        st.image("img/random_forest_concept.svg", caption="随机森林集成学习示意图", use_column_width=True)
    
    # Bagging方法选项卡
    with tabs[1]:
        st.markdown("""
        ## Bagging方法
        
        Bagging (Bootstrap Aggregating) 是一种常用的集成学习方法，名字"Bagging"来自于"Bootstrap Aggregating" (自助抽样聚合)。Bagging的核心思想是降低模型的方差 (Variance)，提高模型的稳定性。
        
        ### 工作原理
        
        1. **自助采样 (Bootstrap Sampling)**: 从原始数据集中有放回地随机抽取多个子数据集 (bootstrap sample)。"有放回"意味着每次抽取的样本，下次抽取时仍然可能被抽到。这样，每个子数据集都和原始数据集大小相近，但样本组成略有不同。
        
        2. **基学习器训练**: 基于每个子数据集，训练一个独立的基学习器 (例如决策树)。
        
        3. **集成 (Aggregating)**: 将所有基学习器的预测结果进行集成。对于分类问题，通常使用投票法 (Voting)，即选择得票最多的类别作为最终预测结果；对于回归问题，通常使用平均法 (Averaging)，即对所有基学习器的预测值取平均。
        """)
        
        st.image("https://miro.medium.com/max/1400/1*LwOBbwGXMZUy8TzUMSjPcg.png", caption="Bagging方法示意图")
        
        st.markdown("""
        ### Bagging的直观理解
        
        Bagging就像是"三个臭皮匠顶个诸葛亮"。每个基学习器就像是一个"臭皮匠"，可能模型能力有限，容易犯错 (高方差)。但通过Bagging将多个"臭皮匠"的预测结果"投票"起来，就相当于进行了"集体决策"，可以降低犯错的概率，得到更可靠、更稳定的预测结果，最终"顶"上一个"诸葛亮" (低方差、高性能的强学习器)。
        
        ### Bagging降低方差的直观解释
        
        想象一下，你要预测明天的天气。如果只问一个气象专家，预测结果可能比较依赖于该专家的个人经验，有一定的偶然性 (高方差)。但如果同时问10个气象专家，并将他们的预测结果综合起来 (例如取平均或投票)，那么最终的预测结果就会更加稳定可靠，不容易受到个别专家预测失误的影响 (低方差)。Bagging的自助采样和集成过程，就类似于"多咨询几个专家，综合决策"的过程，可以有效降低模型的方差，提高模型的稳定性。
        """)
    
    # 随机森林算法选项卡
    with tabs[2]:
        st.markdown("""
        ## 随机森林算法
        
        随机森林 (Random Forest) 是Bagging的一种变体，在Bagging的基础上，进一步引入了特征的随机选择。随机森林以决策树为基学习器，并在决策树的训练过程中引入了随机特征选择。
        
        ### 随机性的两个方面
        
        随机森林的"随机性"体现在两个方面：
        
        1. **样本随机性 (Bagging引入)**: 使用自助采样随机抽取子数据集。
        
        2. **特征随机性 (随机森林特有)**: 在每个节点分裂时，随机选择一部分特征 (而不是考虑所有特征) 进行分裂特征的选择。例如，如果共有 $M$ 个特征，随机森林在每个节点分裂时，会随机选择 $m$ 个特征 ($m < M$)，然后从这 $m$ 个特征中选择最优的分裂特征。通常 $m$ 的取值建议为 $\\sqrt{M}$。
        """)
        
        st.image("https://miro.medium.com/max/1400/1*i0o8mjFfCn-uD79-F1Cqkw.png", caption="随机森林特征随机选择示意图")
        
        st.markdown("""
        ### 特征随机选择的重要性
        
        特征随机选择使得随机森林中的决策树更加"多样化"，进一步降低了模型之间的相关性，使得随机森林的泛化能力更强，更不容易过拟合。
        
        ### 特征随机选择的直观理解
        
        继续用"气象专家预测天气"的例子。随机森林不仅咨询多个气象专家 (Bagging)，而且还要求每个气象专家在预测天气时，只允许参考一部分气象指标 (例如，专家A只能参考温度和湿度，专家B只能参考风速和气压，等等)。这样，每个专家都只能"片面"地看问题，但多个"片面"的预测结果综合起来，反而可能得到更全面、更准确的预测。特征随机选择的目的就是限制单个决策树的能力，避免模型过度依赖于某些强特征，从而提高模型的整体泛化能力。
        
        ### Scikit-learn实现
        
        使用`sklearn.ensemble.RandomForestClassifier`类可以轻松实现随机森林分类器。
        """)
        
        st.code("""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 加载电商用户行为数据集 (假设已预处理完成)
data = pd.read_csv('ecommerce_user_behavior_preprocessed.csv')

# 假设 'label' 列为分类目标变量，其他列为特征变量
X = data.drop('label', axis=1)
y = data['label']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建 RandomForestClassifier 模型
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# 训练模型
rf_model.fit(X_train, y_train)

# 预测测试集
y_pred = rf_model.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"随机森林模型准确率: {accuracy:.4f}")
print("\\n分类报告:\\n", classification_report(y_test, y_pred))

# 特征重要性
feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title("随机森林特征重要性")
plt.show()
        """, language="python")
        
        # 添加特征重要性可视化
        st.image("img/feature_importance_concept.svg", caption="随机森林特征重要性示例", use_column_width=True)
    
    # 优缺点选项卡
    with tabs[3]:
        st.markdown("""
        ## 随机森林的优缺点
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 优点
            
            - **精度高，泛化能力强，不容易过拟合**。随机森林通过集成多个决策树，并引入样本随机性和特征随机性，有效降低了过拟合的风险。
            
            - **能够处理高维数据，不需要进行特征选择**。随机森林在特征选择时引入了随机性，降低了特征维度过高带来的影响。
            
            - **可以评估特征的重要性**。随机森林可以输出每个特征在模型训练过程中的重要性评分，用于特征选择和特征理解。
            
            - **对缺失值和异常值有一定的鲁棒性**。
            
            - **易于并行化，训练速度快**。
            """)
        
        with col2:
            st.markdown("""
            ### 缺点
            
            - **模型可解释性较差**，相对于决策树，随机森林的模型结构更复杂，难以解释。
            
            - **当随机森林中的决策树数量非常大时，模型训练和预测的计算开销会比较大**。
            
            - **对于噪声很大的数据，随机森林可能会过拟合**。
            
            - **对于有很多类别变量的数据，随机森林的表现可能不如单个决策树**。
            """)
    
    # 交互式演示选项卡
    with tabs[4]:
        st.markdown("## 交互式随机森林演示")
        
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
        
        # 模型选择
        model_type = st.radio(
            "选择模型",
            ["随机森林 (Random Forest)", "Bagging决策树 (Bagging Decision Tree)"],
            index=0
        )
        
        # 模型参数设置
        col1, col2 = st.columns(2)
        
        with col1:
            n_estimators = st.slider("基学习器数量", min_value=10, max_value=200, value=100, step=10)
            max_depth = st.slider("最大深度", min_value=1, max_value=20, value=5)
        
        with col2:
            criterion = st.selectbox("分裂准则", ["gini", "entropy"], index=0)
            
            if model_type == "随机森林 (Random Forest)":
                max_features = st.selectbox(
                    "最大特征数",
                    ["sqrt", "log2", None],
                    index=0,
                    format_func=lambda x: "sqrt(n_features)" if x == "sqrt" else "log2(n_features)" if x == "log2" else "所有特征"
                )
            else:
                max_samples = st.slider("样本比例", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
        
        # 训练模型
        if model_type == "随机森林 (Random Forest)":
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                criterion=criterion,
                max_features=max_features,
                random_state=42
            )
        else:
            base_estimator = DecisionTreeClassifier(
                max_depth=max_depth,
                criterion=criterion,
                random_state=42
            )
            model = BaggingClassifier(
                estimator=base_estimator,
                n_estimators=n_estimators,
                max_samples=max_samples,
                random_state=42
            )
        
        model.fit(X_train, y_train)
        
        # 模型评估
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.markdown(f"### 模型准确率: {accuracy:.4f}")
        
        # 交叉验证
        st.markdown("### 交叉验证 (5折)")
        cv_scores = cross_val_score(model, np.vstack((X_train, X_test)), np.hstack((y_train, y_test)), cv=5)
        plot_cross_validation(cv_scores)
        
        # 特征重要性 (仅对随机森林)
        if model_type == "随机森林 (Random Forest)":
            st.markdown("### 特征重要性")
            plot_feature_importance(model, feature_names)
        
        # 混淆矩阵
        st.markdown("### 混淆矩阵")
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, target_names)
        
        # 如果是二分类问题，绘制ROC曲线
        if len(target_names) == 2:
            st.markdown("### ROC曲线")
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plot_roc_curve(fpr, tpr, roc_auc) 

def show_feature_importance():
    """展示随机森林的特征重要性"""
    # 添加特征重要性的可视化图
    st.subheader("特征重要性示例")
    try:
        # 尝试从不同的路径加载SVG文件
        try:
            st.image("img/feature_importance_concept.svg", caption="随机森林特征重要性示例", use_column_width=True)
        except:
            st.image("week4_app/img/feature_importance_concept.svg", caption="随机森林特征重要性示例", use_column_width=True)
    except Exception as e:
        st.error(f"加载特征重要性图片时出错: {e}")
        # 提供文本描述作为备选
        st.markdown("""
        **特征重要性直观解释**:
        
        随机森林可以计算每个特征对模型预测的贡献度，即特征重要性。
        特征重要性通常以条形图展示，横轴表示重要性值，纵轴是特征名称，按重要性降序排列。
        重要性最高的特征对模型的预测影响最大。
        """) 