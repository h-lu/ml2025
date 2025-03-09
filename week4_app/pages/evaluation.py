import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold, 
    GridSearchCV, learning_curve
)

from utils.data_loader import load_dataset
from utils.visualization import (
    plot_confusion_matrix, 
    plot_roc_curve,
    plot_cross_validation,
    plot_grid_search_results,
    plot_learning_curves
)

def show():
    st.markdown('<h2 class="sub-header">模型评估与选择</h2>', unsafe_allow_html=True)
    
    # 创建选项卡
    tabs = st.tabs(["评估指标回顾", "交叉验证", "网格搜索", "评估指标选择", "交互式演示"])
    
    # 评估指标回顾选项卡
    with tabs[0]:
        st.markdown("""
        ## 模型评估指标回顾
        
        在分类问题中，常用的评估指标包括准确率、精确率、召回率、F1-score、AUC-ROC等。
        
        ### 混淆矩阵
        
        混淆矩阵是评估分类模型性能的基础，它展示了模型预测结果与真实标签之间的关系。对于二分类问题，混淆矩阵包含四个值：
        
        - **真正例 (True Positive, TP)**: 实际为正例，预测为正例
        - **假正例 (False Positive, FP)**: 实际为负例，预测为正例
        - **真负例 (True Negative, TN)**: 实际为负例，预测为负例
        - **假负例 (False Negative, FN)**: 实际为正例，预测为负例
        """)
        
        st.image("https://miro.medium.com/max/1400/1*fxiTNIgOyvAombPJx5KGeA.png", caption="混淆矩阵")
        
        st.markdown("""
        ### 常用评估指标
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 准确率 (Accuracy)
            
            准确率是所有预测正确的样本占总样本的比例。
            
            $Accuracy = \\frac{TP + TN}{TP + TN + FP + FN}$
            
            #### 精确率 (Precision)
            
            精确率是预测为正例的样本中，真正的正例的比例。
            
            $Precision = \\frac{TP}{TP + FP}$
            
            #### 召回率 (Recall)
            
            召回率是真正的正例中，被预测为正例的比例。
            
            $Recall = \\frac{TP}{TP + FN}$
            """)
        
        with col2:
            st.markdown("""
            #### F1-score
            
            F1-score是精确率和召回率的调和平均值。
            
            $F1 = 2 \\times \\frac{Precision \\times Recall}{Precision + Recall}$
            
            #### AUC-ROC
            
            AUC-ROC是ROC曲线下的面积，ROC曲线描述了在不同阈值下，真正例率(TPR)和假正例率(FPR)之间的关系。
            
            $TPR = \\frac{TP}{TP + FN}$
            
            $FPR = \\frac{FP}{FP + TN}$
            
            AUC值越大，模型性能越好。
            """)
        
        st.image("https://miro.medium.com/max/1400/1*pk05QGzoWhCgRiiFbz-oKQ.png", caption="ROC曲线示例")
        
        st.markdown("""
        ### 2. 混淆矩阵（Confusion Matrix）

        混淆矩阵是分类问题中评估模型性能的重要工具，它可以帮助我们理解模型在各个类别上的表现。

        **混淆矩阵的组成**:
        - 真正例（True Positive, TP）：模型预测为正例，实际也是正例
        - 假正例（False Positive, FP）：模型预测为正例，实际是负例
        - 真负例（True Negative, TN）：模型预测为负例，实际也是负例
        - 假负例（False Negative, FN）：模型预测为负例，实际是正例
        """)
        
        # 混淆矩阵可视化
        try:
            # 尝试从绝对路径和相对路径加载图像
            try:
                st.image("img/extra/confusion_matrix.svg", caption="混淆矩阵示例", use_column_width=True)
            except:
                st.image("week4_app/img/extra/confusion_matrix.svg", caption="混淆矩阵示例", use_column_width=True)
        except Exception as e:
            st.error(f"加载混淆矩阵示意图时出错: {e}")
            # 显示文本描述作为备选
            st.markdown("""
            **混淆矩阵示例**:
            
            |  | 预测负例 | 预测正例 |
            |---|----------|----------|
            | **实际负例** | 85 (TN) | 15 (FP) |
            | **实际正例** | 10 (FN) | 90 (TP) |
            """)
    
    # 交叉验证选项卡
    with tabs[1]:
        st.markdown("""
        ## 交叉验证 (Cross-Validation)
        
        交叉验证是一种评估模型性能的方法，它通过将数据集划分为多个子集，使用其中一部分作为测试集，其余部分作为训练集，多次训练和评估模型，最终取平均值作为模型性能的估计。
        
        ### K折交叉验证 (K-Fold Cross-Validation)
        
        K折交叉验证是最常用的交叉验证方法，它将数据集划分为K个大小相近的子集（折叠），然后进行K次训练和评估，每次使用其中一个子集作为测试集，其余K-1个子集作为训练集。
        """)
        
        # 交叉验证部分
        st.markdown("### 3. 交叉验证")
        st.markdown("""
        交叉验证是一种评估模型性能的方法，避免训练和测试数据分割的偶然性带来的影响。最常用的是k折交叉验证（k-fold cross-validation）。
        
        **交叉验证的步骤**:
        1. 将数据集分成k个大小相近的子集（fold）
        2. 每次使用k-1个子集作为训练集，余下的一个子集作为验证集
        3. 重复k次，每个子集都有机会作为验证集
        4. 将k次的结果取平均值作为最终性能指标
        """)
        
        # 交叉验证示意图
        try:
            # 尝试从不同的路径加载SVG文件
            try:
                st.image("img/cross_validation_concept.svg", caption="5折交叉验证示意图", use_column_width=True)
            except:
                st.image("week4_app/img/cross_validation_concept.svg", caption="5折交叉验证示意图", use_column_width=True)
        except Exception as e:
            st.error(f"加载交叉验证示意图时出错: {e}")
            # 显示文本描述作为备选
            st.markdown("""
            5折交叉验证过程**:
            
            1. 将数据集随机分为5份大小相当的子集
            2. 第1次：用第1份作为验证集，其余4份作为训练集
            3. 第2次：用第2份作为验证集，其余4份作为训练集
            4. ...依此类推，直到第5次
            5. 最终模型性能评估是5次评估结果的平均值
            """)
        
        st.markdown("""
        ### 交叉验证的优点
        
        - **更可靠的性能评估**: 通过多次训练和评估，减少了单次划分带来的偶然性。
        - **充分利用数据**: 每个样本都会被用作训练和测试，充分利用了有限的数据。
        - **避免过拟合**: 通过多次评估，可以更好地检测模型是否过拟合。
        
        ### Scikit-learn实现
        
        使用`sklearn.model_selection.cross_val_score`函数可以轻松实现交叉验证。
        """)
        
        st.code("""
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# 创建模型
model = DecisionTreeClassifier(max_depth=5, random_state=42)

# 进行5折交叉验证
cv_scores = cross_val_score(model, X, y, cv=5)

# 输出交叉验证结果
print(f"交叉验证得分: {cv_scores}")
print(f"平均得分: {cv_scores.mean():.4f}, 标准差: {cv_scores.std():.4f}")
        """, language="python")
    
    # 网格搜索选项卡
    with tabs[2]:
        st.markdown("""
        ## 网格搜索 (Grid Search)
        
        网格搜索是一种超参数调优方法，它通过穷举搜索指定的参数空间，找到最优的参数组合。
        
        ### 工作原理
        
        1. 定义参数网格，即要搜索的参数及其可能的取值。
        2. 对参数网格中的每个参数组合，使用交叉验证评估模型性能。
        3. 选择性能最好的参数组合作为最终的模型参数。
        
        ### Scikit-learn实现
        
        使用`sklearn.model_selection.GridSearchCV`类可以轻松实现网格搜索。
        """)
        
        st.code("""
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 创建模型
model = RandomForestClassifier(random_state=42)

# 创建网格搜索对象
grid_search = GridSearchCV(
    model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)

# 进行网格搜索
grid_search.fit(X_train, y_train)

# 输出最佳参数和得分
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳得分: {grid_search.best_score_:.4f}")

# 使用最佳参数的模型进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
        """, language="python")
        
        st.markdown("""
        ### 网格搜索的优缺点
        
        #### 优点
        
        - **全面搜索**: 可以全面搜索参数空间，找到最优参数组合。
        - **自动化**: 自动化了参数调优过程，减少了人工调参的工作量。
        - **结合交叉验证**: 结合交叉验证，可以得到更可靠的参数评估。
        
        #### 缺点
        
        - **计算开销大**: 当参数空间很大时，计算开销会非常大。
        - **可能错过最优解**: 如果参数取值设置不当，可能会错过真正的最优解。
        """)
    
    # 评估指标选择选项卡
    with tabs[3]:
        st.markdown("""
        ## 分类模型评估指标的选择和应用场景
        
        选择合适的评估指标，需要根据具体的业务目标和问题类型来决定。不同的评估指标关注模型的不同方面，适用于不同的应用场景。
        """)
        
        st.markdown("""
        ### 准确率 (Accuracy)
        
        **最常用的评估指标之一**，表示模型预测正确的样本比例。**适用于类别分布均衡的分类问题**。
        
        #### 适用场景
        
        - 手写数字识别、图像分类等问题，如果每个类别的样本数量相差不大，可以使用准确率作为主要评估指标。
        
        #### 局限性
        
        - 当类别分布不均衡时，准确率可能会产生误导。例如，如果在一个疾病预测问题中，99%的样本都是健康人，模型如果将所有人都预测为健康，也能达到99%的准确率，但这显然不是一个好的模型。
        """)
        
        st.markdown("""
        ### 精确率 (Precision) 和 召回率 (Recall)
        
        **适用于类别分布不均衡的分类问题**。精确率关注"预测为正例的样本中，有多少是真正的正例"，召回率关注"真正的正例样本中，有多少被模型预测为正例"。
        
        #### 精确率的应用场景
        
        - **当更关注"预测为正例的准确性"时**，例如，在垃圾邮件识别中，我们更关注"被模型判断为垃圾邮件的邮件，有多少是真正的垃圾邮件"，因为如果将正常邮件误判为垃圾邮件，可能会造成用户的重要信息丢失，误判的代价较高。**此时，我们希望提高精确率，降低误判率 (FP)。**
        
        #### 召回率的应用场景
        
        - **当更关注"对正例的识别能力"时**，例如，在疾病诊断中，我们更关注"真正的病人中，有多少被模型诊断出来"，因为如果漏诊病人，可能会延误治疗，造成更严重的后果，漏诊的代价较高。**此时，我们希望提高召回率，降低漏诊率 (FN)。**
        """)
        
        st.markdown("""
        ### F1-score
        
        **精确率和召回率的调和平均值，综合考虑了精确率和召回率**。**适用于类别分布不均衡，且希望平衡精确率和召回率的场景**。
        
        #### 适用场景
        
        - 在欺诈交易检测、用户流失预测等问题中，我们既希望尽可能准确地识别出欺诈交易或潜在流失用户 (提高精确率)，也希望尽可能全面地覆盖所有欺诈交易或潜在流失用户 (提高召回率)，此时可以使用F1-score作为综合评估指标。
        """)
        
        st.markdown("""
        ### AUC-ROC (Area Under the ROC Curve)
        
        **适用于二分类问题，特别是当需要权衡不同阈值下的模型性能时**。ROC曲线描述了在不同阈值下，模型的真正例率 (TPR, 召回率) 和假正例率 (FPR) 之间的关系。**AUC值是ROC曲线下的面积，AUC值越大，模型性能越好**。
        
        #### 适用场景
        
        - AUC-ROC关注的是模型对正负样本的排序能力，**对类别分布不均衡的情况不敏感**，因此在类别不均衡问题中也经常使用。
        - 在风险评分、信用评级等需要对样本进行排序的问题中，AUC-ROC是一个很好的评估指标。
        """)
        
        st.markdown("""
        ### 模型评估指标选择总结
        
        没有"万能"的评估指标，选择合适的评估指标需要根据具体的业务场景和问题目标来决定。**理解各种评估指标的含义和适用场景，才能更好地评估模型性能，并根据评估结果优化模型。**
        """)
    
    # 交互式演示选项卡
    with tabs[4]:
        st.markdown("## 交互式模型评估与选择演示")
        
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
            ["决策树 (Decision Tree)", "随机森林 (Random Forest)"],
            index=0
        )
        
        # 评估方法选择
        evaluation_method = st.radio(
            "选择评估方法",
            ["交叉验证", "网格搜索", "学习曲线"],
            index=0
        )
        
        # 根据评估方法显示不同的参数设置
        if evaluation_method == "交叉验证":
            n_folds = st.slider("折叠数", min_value=2, max_value=10, value=5)
            
            if model_type == "决策树 (Decision Tree)":
                max_depth = st.slider("最大深度", min_value=1, max_value=20, value=3)
                criterion = st.selectbox("分裂准则", ["gini", "entropy"], index=0)
                
                model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    criterion=criterion,
                    random_state=42
                )
            else:
                n_estimators = st.slider("基学习器数量", min_value=10, max_value=200, value=100, step=10)
                max_depth = st.slider("最大深度", min_value=1, max_value=20, value=5)
                
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
            
            # 执行交叉验证
            cv_scores = cross_val_score(
                model, 
                np.vstack((X_train, X_test)), 
                np.hstack((y_train, y_test)), 
                cv=n_folds
            )
            
            # 显示交叉验证结果
            st.markdown(f"### 交叉验证结果 ({n_folds}折)")
            st.markdown(f"**平均得分**: {cv_scores.mean():.4f}, **标准差**: {cv_scores.std():.4f}")
            
            # 可视化交叉验证结果
            plot_cross_validation(cv_scores)
            
        elif evaluation_method == "网格搜索":
            if model_type == "决策树 (Decision Tree)":
                # 决策树参数网格
                max_depth_values = st.multiselect(
                    "最大深度",
                    options=list(range(1, 21)),
                    default=[3, 5, 7, 10]
                )
                
                criterion_values = st.multiselect(
                    "分裂准则",
                    options=["gini", "entropy"],
                    default=["gini", "entropy"]
                )
                
                min_samples_split_values = st.multiselect(
                    "最小分裂样本数",
                    options=[2, 5, 10, 20],
                    default=[2, 5]
                )
                
                param_grid = {
                    'max_depth': max_depth_values,
                    'criterion': criterion_values,
                    'min_samples_split': min_samples_split_values
                }
                
                base_model = DecisionTreeClassifier(random_state=42)
                
            else:
                # 随机森林参数网格
                n_estimators_values = st.multiselect(
                    "基学习器数量",
                    options=[10, 50, 100, 200],
                    default=[50, 100]
                )
                
                max_depth_values = st.multiselect(
                    "最大深度",
                    options=list(range(1, 21)) + [None],
                    default=[5, 10, None]
                )
                
                max_features_values = st.multiselect(
                    "最大特征数",
                    options=["sqrt", "log2", None],
                    default=["sqrt", "log2"]
                )
                
                param_grid = {
                    'n_estimators': n_estimators_values,
                    'max_depth': max_depth_values,
                    'max_features': max_features_values
                }
                
                base_model = RandomForestClassifier(random_state=42)
            
            # 执行网格搜索
            n_folds = st.slider("交叉验证折叠数", min_value=2, max_value=10, value=3)
            
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=n_folds,
                scoring='accuracy',
                return_train_score=True
            )
            
            with st.spinner("正在进行网格搜索，请稍候..."):
                grid_search.fit(X_train, y_train)
            
            # 显示网格搜索结果
            st.markdown("### 网格搜索结果")
            st.markdown(f"**最佳参数**: {grid_search.best_params_}")
            st.markdown(f"**最佳得分**: {grid_search.best_score_:.4f}")
            
            # 使用最佳参数的模型进行预测
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.markdown(f"**测试集准确率**: {accuracy:.4f}")
            
            # 可视化网格搜索结果
            if model_type == "决策树 (Decision Tree)":
                if len(max_depth_values) > 1:
                    st.markdown("### 最大深度参数影响")
                    plot_grid_search_results(grid_search.cv_results_, 'max_depth')
            else:
                if len(n_estimators_values) > 1:
                    st.markdown("### 基学习器数量参数影响")
                    plot_grid_search_results(grid_search.cv_results_, 'n_estimators')
            
        else:  # 学习曲线
            if model_type == "决策树 (Decision Tree)":
                max_depth = st.slider("最大深度", min_value=1, max_value=20, value=3)
                criterion = st.selectbox("分裂准则", ["gini", "entropy"], index=0)
                
                model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    criterion=criterion,
                    random_state=42
                )
            else:
                n_estimators = st.slider("基学习器数量", min_value=10, max_value=200, value=100, step=10)
                max_depth = st.slider("最大深度", min_value=1, max_value=20, value=5)
                
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
            
            # 执行学习曲线分析
            n_folds = st.slider("交叉验证折叠数", min_value=2, max_value=10, value=3)
            
            # 设置训练集大小
            train_sizes = np.linspace(0.1, 1.0, 10)
            
            with st.spinner("正在计算学习曲线，请稍候..."):
                train_sizes, train_scores, test_scores = learning_curve(
                    model, 
                    np.vstack((X_train, X_test)), 
                    np.hstack((y_train, y_test)), 
                    train_sizes=train_sizes,
                    cv=n_folds,
                    scoring='accuracy',
                    n_jobs=-1
                )
            
            # 可视化学习曲线
            st.markdown("### 学习曲线")
            plot_learning_curves(train_scores, test_scores, train_sizes)
            
            # 解释学习曲线
            train_mean = np.mean(train_scores, axis=1)[-1]
            test_mean = np.mean(test_scores, axis=1)[-1]
            gap = train_mean - test_mean
            
            st.markdown("### 学习曲线分析")
            
            if gap > 0.1:
                st.markdown(f"""
                **高方差 (过拟合)**: 训练集得分 ({train_mean:.4f}) 明显高于测试集得分 ({test_mean:.4f})，差距为 {gap:.4f}。
                
                **建议**:
                - 收集更多训练数据
                - 减小模型复杂度 (例如，减小树的深度，增加min_samples_split或min_samples_leaf)
                - 尝试特征选择或特征降维
                """)
            elif test_mean < 0.7:
                st.markdown(f"""
                **高偏差 (欠拟合)**: 测试集得分较低 ({test_mean:.4f})，且训练集得分 ({train_mean:.4f}) 也不高。
                
                **建议**:
                - 增加模型复杂度 (例如，增加树的深度，增加n_estimators)
                - 添加更多特征或特征工程
                - 考虑使用其他算法
                """)
            else:
                st.markdown(f"""
                **良好拟合**: 训练集得分 ({train_mean:.4f}) 和测试集得分 ({test_mean:.4f}) 都较高，且差距较小 ({gap:.4f})。
                
                **建议**:
                - 当前模型表现良好，可以使用此模型进行预测
                - 可以尝试微调参数进一步提升性能
                """) 