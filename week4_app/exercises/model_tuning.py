import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from utils.data_loader import load_dataset
from utils.visualization import plot_grid_search_results, plot_learning_curves

def show_model_tuning():
    """显示超参数调优的各种方法和结果"""
    st.markdown("## 高级练习3: 超参数调优与模型评估")
    
    st.markdown("""
    ### 目标
    
    深入理解模型调优方法，学习如何使用网格搜索和交叉验证进行超参数调优，以及如何解读学习曲线和模型评估结果。
    
    ### 任务
    
    1. 加载分类数据集
    2. 设置参数网格并使用网格搜索进行超参数调优
    3. 分析模型在不同参数组合下的性能
    4. 绘制并分析学习曲线，诊断过拟合和欠拟合问题
    5. 使用最佳参数训练最终模型并进行评估
    """)
    
    # 数据集选择
    dataset_name = st.selectbox(
        "选择数据集",
        ["iris", "wine", "breast_cancer", "ecommerce"],
        index=1,
        key="model_tuning_dataset"
    )
    
    # 加载数据集
    X_train, X_test, y_train, y_test, feature_names, target_names = load_dataset(
        dataset_name, test_size=0.2, random_state=42
    )
    
    # 超参数选择
    st.markdown("### 超参数选择")
    st.markdown("""
    选择要调优的超参数范围。随机森林的主要超参数包括：
    - n_estimators: 树的数量
    - max_depth: 树的最大深度
    - min_samples_split: 内部节点分裂所需的最小样本数
    - min_samples_leaf: 叶节点所需的最小样本数
    - max_features: 分裂时考虑的特征数量
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators_min = st.number_input("n_estimators 最小值", min_value=10, max_value=500, value=50, step=10)
        n_estimators_max = st.number_input("n_estimators 最大值", min_value=10, max_value=500, value=200, step=10)
        n_estimators_step = st.number_input("n_estimators 步长", min_value=10, max_value=100, value=50, step=10)
        
        max_depth_values = st.multiselect(
            "max_depth 值",
            options=[None] + list(range(1, 31)),
            default=[5, 10, None]
        )
        
    with col2:
        min_samples_split_values = st.multiselect(
            "min_samples_split 值",
            options=list(range(2, 21)),
            default=[2, 5, 10]
        )
        
        min_samples_leaf_values = st.multiselect(
            "min_samples_leaf 值",
            options=list(range(1, 11)),
            default=[1, 2, 4]
        )
        
        max_features_values = st.multiselect(
            "max_features 值",
            options=["sqrt", "log2", None],
            default=["sqrt", "log2"]
        )
    
    # 交叉验证设置
    cv_folds = st.slider("交叉验证折数", min_value=2, max_value=10, value=5)
    
    # 评分指标选择
    scoring = st.selectbox(
        "评分指标",
        ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"],
        index=0
    )
    
    if st.button("运行超参数调优", key="run_hyperparameter_tuning"):
        with st.spinner("正在进行网格搜索，这可能需要几分钟..."):
            # 创建参数网格
            n_estimators_range = list(range(n_estimators_min, n_estimators_max + 1, n_estimators_step))
            
            param_grid = {
                'n_estimators': n_estimators_range,
                'max_depth': max_depth_values,
                'min_samples_split': min_samples_split_values,
                'min_samples_leaf': min_samples_leaf_values,
                'max_features': max_features_values
            }
            
            # 创建随机森林模型
            rf = RandomForestClassifier(random_state=42)
            
            # 创建网格搜索对象
            grid_search = GridSearchCV(
                estimator=rf,
                param_grid=param_grid,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
                return_train_score=True
            )
            
            # 执行网格搜索
            grid_search.fit(X_train, y_train)
            
            # 显示最佳参数
            st.markdown("### 网格搜索结果")
            st.markdown(f"**最佳参数**: {grid_search.best_params_}")
            st.markdown(f"**最佳交叉验证得分**: {grid_search.best_score_:.4f}")
            
            # 将网格搜索结果转换为DataFrame
            results = pd.DataFrame(grid_search.cv_results_)
            
            # 显示前10个最佳结果
            st.markdown("#### 前10个最佳参数组合")
            
            # 对结果进行排序
            sorted_results = results.sort_values(by="rank_test_score").head(10)
            
            # 提取关键列
            display_columns = ['rank_test_score', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']
            param_columns = [col for col in sorted_results.columns if col.startswith('param_')]
            
            display_results = sorted_results[display_columns + param_columns].copy()
            
            # 重命名列使其更易读
            display_results = display_results.rename(columns={
                'rank_test_score': '排名',
                'mean_test_score': '测试集平均分数',
                'std_test_score': '测试集标准差',
                'mean_train_score': '训练集平均分数',
                'std_train_score': '训练集标准差'
            })
            
            # 对参数列进行重命名
            for col in param_columns:
                new_col = col.replace('param_', '')
                display_results = display_results.rename(columns={col: new_col})
            
            st.dataframe(display_results)
            
            # 可视化单个参数的影响
            st.markdown("### 参数影响分析")
            
            param_to_analyze = st.selectbox(
                "选择要分析的参数",
                options=[p.replace('param_', '') for p in param_columns],
                index=0
            )
            
            # 使用自定义的可视化函数绘制网格搜索结果
            st.markdown(f"#### {param_to_analyze} 参数对模型性能的影响")
            plot_grid_search_results(grid_search.cv_results_, f'param_{param_to_analyze}')
            
            # 使用最佳参数创建模型
            best_rf = RandomForestClassifier(**grid_search.best_params_, random_state=42)
            
            # 绘制学习曲线
            st.markdown("### 学习曲线分析")
            
            with st.spinner("正在计算学习曲线..."):
                train_sizes = np.linspace(0.1, 1.0, 10)
                
                train_sizes, train_scores, test_scores = learning_curve(
                    best_rf,
                    np.vstack((X_train, X_test)),
                    np.hstack((y_train, y_test)),
                    train_sizes=train_sizes,
                    cv=cv_folds,
                    scoring=scoring,
                    n_jobs=-1
                )
                
                # 使用自定义的可视化函数绘制学习曲线
                plot_learning_curves(train_scores, test_scores, train_sizes)
                
                # 计算最终的训练集和测试集得分
                train_mean = np.mean(train_scores, axis=1)[-1]
                test_mean = np.mean(test_scores, axis=1)[-1]
                gap = train_mean - test_mean
                
                # 学习曲线诊断
                st.markdown("#### 学习曲线诊断")
                
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
            
            # 使用最佳参数在完整训练集上训练最终模型
            with st.spinner("正在训练最终模型..."):
                best_rf.fit(X_train, y_train)
                
                # 在测试集上评估
                y_pred = best_rf.predict(X_test)
                
                # 计算性能指标
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # 输出性能指标
                st.markdown("### 最终模型性能")
                st.markdown(f"**准确率**: {accuracy:.4f}")
                st.markdown(f"**F1分数**: {f1:.4f}")
                
                # 显示分类报告
                report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.markdown("#### 分类报告")
                st.dataframe(report_df)
                
                # 绘制混淆矩阵
                cm = confusion_matrix(y_test, y_pred)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues',
                    xticklabels=target_names,
                    yticklabels=target_names,
                    ax=ax
                )
                ax.set_xlabel('预测标签')
                ax.set_ylabel('真实标签')
                ax.set_title('混淆矩阵')
                
                st.pyplot(fig)
                
                # 显示特征重要性
                if len(feature_names) <= 30:  # 仅对特征数量合理的情况进行可视化
                    feature_importances = pd.DataFrame({
                        'feature': feature_names,
                        'importance': best_rf.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    st.markdown("#### 特征重要性")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(
                        x='importance', 
                        y='feature', 
                        data=feature_importances, 
                        ax=ax
                    )
                    ax.set_title('特征重要性')
                    ax.set_xlabel('重要性')
                    ax.set_ylabel('特征')
                    
                    st.pyplot(fig)
    
    # 学习曲线部分
    st.markdown("### 学习曲线分析")
    st.markdown("""
    学习曲线展示了训练集大小对模型性能的影响，能帮助诊断模型是否过拟合或欠拟合。
    """)
    
    # 添加学习曲线概念图展示
    try:
        # 尝试从不同的路径加载SVG文件
        try:
            st.image("img/extra/learning_curve.svg", caption="学习曲线概念图", use_column_width=True)
        except:
            st.image("week4_app/img/extra/learning_curve.svg", caption="学习曲线概念图", use_column_width=True)
    except Exception as e:
        st.error(f"加载学习曲线概念图时出错: {e}")
        # 提供文本说明作为备选
        st.markdown("""
        **学习曲线解释**:
        - 随着训练样本增加，训练集得分下降，验证集得分上升
        - 当两条曲线逐渐接近且平稳时，表明模型达到了较好的拟合状态
        - 曲线间的差距大表示过拟合，两者都低表示欠拟合
        """)
    
    # 提供参考代码
    with st.expander("查看参考代码"):
        st.code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 加载数据集
data = load_wine()
X, y = data.data, data.target
feature_names = data.feature_names
target_names = data.target_names

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# 创建随机森林模型
rf = RandomForestClassifier(random_state=42)

# 创建网格搜索对象
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    return_train_score=True
)

# 执行网格搜索
grid_search.fit(X_train, y_train)

# 输出最佳参数
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# 将网格搜索结果转换为DataFrame
results = pd.DataFrame(grid_search.cv_results_)

# 对结果进行排序并显示前10个最佳结果
sorted_results = results.sort_values(by="rank_test_score").head(10)
display_columns = ['rank_test_score', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']
param_columns = [col for col in sorted_results.columns if col.startswith('param_')]
print(sorted_results[display_columns + param_columns])

# 可视化参数影响
# 例如，分析n_estimators参数的影响
param_name = 'param_n_estimators'
param_values = results[param_name].astype(str).unique()
mean_scores = []
std_scores = []

for value in param_values:
    value_results = results[results[param_name].astype(str) == value]
    mean_scores.append(value_results['mean_test_score'].mean())
    std_scores.append(value_results['mean_test_score'].std())

plt.figure(figsize=(10, 6))
plt.errorbar(param_values, mean_scores, yerr=std_scores, marker='o')
plt.xlabel('n_estimators')
plt.ylabel('Mean Test Score')
plt.title('Effect of n_estimators on Model Performance')
plt.grid(True)
plt.show()

# 使用最佳参数创建模型
best_rf = RandomForestClassifier(**grid_search.best_params_, random_state=42)

# 计算学习曲线
train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes, train_scores, test_scores = learning_curve(
    best_rf,
    X,
    y,
    train_sizes=train_sizes,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# 计算均值和标准差
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# 绘制学习曲线
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# 在完整训练集上训练最终模型
best_rf.fit(X_train, y_train)

# 在测试集上评估
y_pred = best_rf.predict(X_test)

# 计算性能指标
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# 显示特征重要性
feature_importances = pd.DataFrame({
    'feature': feature_names,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
        """, language="python")
    
    st.markdown("""
    ### 思考问题
    
    1. 在超参数调优过程中，如何平衡搜索空间的广度和计算时间的限制？
    2. 学习曲线如何帮助我们诊断模型的偏差和方差问题？
    3. 在实际应用中，如何选择合适的评估指标，以及如何解释不同评估指标之间的权衡？
    """) 