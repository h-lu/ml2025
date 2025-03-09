import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils.data_loader import load_dataset
from utils.helpers import generate_ecommerce_data

def show_feature_importance_analysis():
    st.markdown("## 高级练习2: 特征重要性分析与特征选择")
    
    st.markdown("""
    ### 目标
    
    深入分析模型特征重要性，掌握不同的特征重要性分析方法，并进行基于特征重要性的特征选择。
    
    ### 任务
    
    1. 加载数据集并探索特征
    2. 训练随机森林模型，计算不同类型的特征重要性
    3. 比较和可视化不同的特征重要性计算方法
    4. 基于特征重要性进行特征选择，优化模型性能
    """)
    
    # 数据集选择
    dataset_option = st.radio(
        "选择数据集",
        ["内置数据集", "生成电商数据集"],
        index=0,
        key="feature_importance_dataset_option"
    )
    
    if dataset_option == "内置数据集":
        dataset_name = st.selectbox(
            "选择内置数据集",
            ["breast_cancer", "wine"],
            index=0,
            key="feature_importance_dataset"
        )
        
        # 加载数据集
        X_train, X_test, y_train, y_test, feature_names, target_names = load_dataset(
            dataset_name, test_size=0.2, random_state=42
        )
        
        # 确保X_train和X_test是DataFrame
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
    else:
        n_samples = st.slider(
            "样本数量", 
            min_value=100, 
            max_value=5000, 
            value=1000,
            step=100,
            key="n_samples"
        )
        
        # 生成电商数据集
        df = generate_ecommerce_data(n_samples=n_samples, random_state=42)
        
        # 显示数据集预览
        st.dataframe(df.head())
        
        # 准备特征和目标变量
        X = df.drop(['user_id', 'high_value_user'], axis=1)
        y = df['high_value_user']
        feature_names = X.columns.tolist()
        target_names = ['低价值用户', '高价值用户']
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 这里X_train和X_test已经是DataFrame
        X_train_df = X_train
        X_test_df = X_test
    
    # 特征重要性计算方法
    importance_methods = st.multiselect(
        "选择特征重要性计算方法",
        ["基于树的特征重要性 (Tree-based)", "基于排列的特征重要性 (Permutation)"],
        default=["基于树的特征重要性 (Tree-based)"],
        key="importance_methods"
    )
    
    # 模型参数
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider(
            "树的数量", 
            min_value=10, 
            max_value=200, 
            value=100,
            step=10,
            key="n_estimators_fi"
        )
    
    with col2:
        max_depth = st.slider(
            "最大深度", 
            min_value=2, 
            max_value=30, 
            value=10,
            key="max_depth_fi"
        )
    
    # 特征选择阈值
    importance_threshold = st.slider(
        "特征重要性阈值 (百分比)", 
        min_value=0, 
        max_value=100, 
        value=80,
        key="importance_threshold"
    )
    
    if st.button("运行特征重要性分析", key="run_feature_importance"):
        with st.spinner("正在计算特征重要性..."):
            # 创建并训练随机森林模型
            rf_model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            
            rf_model.fit(X_train_df, y_train)
            
            # 评估基础模型性能
            y_pred = rf_model.predict(X_test_df)
            base_accuracy = accuracy_score(y_test, y_pred)
            
            st.markdown(f"### 基础模型准确率: {base_accuracy:.4f}")
            
            # 计算并展示不同类型的特征重要性
            importance_results = {}
            
            if "基于树的特征重要性 (Tree-based)" in importance_methods:
                # 计算基于树的特征重要性
                tree_importances = pd.DataFrame({
                    'feature': feature_names,
                    'importance': rf_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                importance_results["基于树的特征重要性 (Tree-based)"] = tree_importances
                
                st.markdown("### 基于树的特征重要性")
                st.dataframe(tree_importances)
                
                # 可视化基于树的特征重要性
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(
                    x='importance', 
                    y='feature', 
                    data=tree_importances.head(15),
                    ax=ax
                )
                ax.set_title('基于树的特征重要性 (前15个特征)')
                ax.set_xlabel('重要性')
                ax.set_ylabel('特征')
                
                st.pyplot(fig)
            
            if "基于排列的特征重要性 (Permutation)" in importance_methods:
                # 计算基于排列的特征重要性
                perm_importance = permutation_importance(
                    rf_model, X_test_df, y_test, n_repeats=10, random_state=42
                )
                
                perm_importances = pd.DataFrame({
                    'feature': feature_names,
                    'importance': perm_importance.importances_mean
                }).sort_values('importance', ascending=False)
                
                importance_results["基于排列的特征重要性 (Permutation)"] = perm_importances
                
                st.markdown("### 基于排列的特征重要性")
                st.dataframe(perm_importances)
                
                # 可视化基于排列的特征重要性
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(
                    x='importance', 
                    y='feature', 
                    data=perm_importances.head(15),
                    ax=ax
                )
                ax.set_title('基于排列的特征重要性 (前15个特征)')
                ax.set_xlabel('重要性')
                ax.set_ylabel('特征')
                
                st.pyplot(fig)
            
            # 如果有多个特征重要性计算方法，比较它们的结果
            if len(importance_methods) > 1:
                st.markdown("### 不同特征重要性方法的比较")
                
                # 合并不同方法的特征重要性
                comparison_df = pd.DataFrame({'feature': feature_names})
                
                for method, result in importance_results.items():
                    # 归一化特征重要性
                    result_norm = result.copy()
                    result_norm['importance'] = result_norm['importance'] / result_norm['importance'].max()
                    
                    comparison_df = comparison_df.merge(
                        result_norm,
                        on='feature',
                        suffixes=('', f'_{method}')
                    )
                
                # 显示比较结果
                st.dataframe(comparison_df.sort_values('importance_基于树的特征重要性 (Tree-based)', ascending=False))
                
                # 可视化比较结果
                fig, ax = plt.subplots(figsize=(10, 6))
                
                x = np.arange(len(feature_names))
                width = 0.35
                
                for i, (method, result) in enumerate(importance_results.items()):
                    # 归一化重要性
                    norm_importances = result['importance'] / result['importance'].max()
                    
                    # 按照特征名称排序，以便两种方法可以并排显示
                    sorted_indices = np.argsort(feature_names)
                    
                    ax.bar(
                        x + i*width, 
                        norm_importances[sorted_indices], 
                        width, 
                        label=method
                    )
                
                ax.set_xticks(x + width/2)
                ax.set_xticklabels([feature_names[i] for i in sorted_indices], rotation=90)
                ax.legend()
                ax.set_title('不同特征重要性方法比较 (归一化)')
                
                st.pyplot(fig)
            
            # 基于特征重要性进行特征选择
            st.markdown("### 基于特征重要性的特征选择")
            
            # 使用基于树的特征重要性（如果计算了的话），否则使用基于排列的特征重要性
            if "基于树的特征重要性 (Tree-based)" in importance_methods:
                selected_importance = importance_results["基于树的特征重要性 (Tree-based)"]
            else:
                selected_importance = importance_results["基于排列的特征重要性 (Permutation)"]
            
            # 计算累计重要性的百分比
            total_importance = selected_importance['importance'].sum()
            selected_importance['cumulative_importance'] = selected_importance['importance'].cumsum() / total_importance * 100
            
            # 根据阈值选择特征
            selected_features = selected_importance[selected_importance['cumulative_importance'] <= importance_threshold]['feature'].tolist()
            
            # 如果没有特征被选中，至少选择最重要的一个特征
            if not selected_features:
                selected_features = [selected_importance.iloc[0]['feature']]
            
            st.markdown(f"选择了 {len(selected_features)}/{len(feature_names)} 个特征，累计重要性阈值: {importance_threshold}%")
            st.write("选中的特征:", selected_features)
            
            # 可视化累计重要性
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(
                range(1, len(selected_importance) + 1), 
                selected_importance['cumulative_importance'],
                'o-'
            )
            ax.axhline(y=importance_threshold, color='r', linestyle='--', label=f'{importance_threshold}% 阈值')
            ax.set_xlabel('特征数量')
            ax.set_ylabel('累计重要性 (%)')
            ax.set_title('特征累计重要性')
            ax.legend()
            ax.grid(True)
            
            st.pyplot(fig)
            
            # 使用选定的特征训练新模型
            X_train_selected = X_train_df[selected_features]
            X_test_selected = X_test_df[selected_features]
            
            rf_model_selected = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            
            rf_model_selected.fit(X_train_selected, y_train)
            
            # 评估新模型性能
            y_pred_selected = rf_model_selected.predict(X_test_selected)
            selected_accuracy = accuracy_score(y_test, y_pred_selected)
            
            st.markdown(f"### 特征选择后的模型准确率: {selected_accuracy:.4f}")
            
            # 比较原始模型和特征选择后的模型
            st.markdown("### 模型比较")
            
            comparison_data = {
                "模型": ["所有特征", "选择特征"],
                "使用特征数": [len(feature_names), len(selected_features)],
                "准确率": [base_accuracy, selected_accuracy]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df)
            
            # 可视化模型比较
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x="模型", y="准确率", data=comparison_df, ax=ax)
            ax.set_ylim(0, 1)
            ax.set_title("特征选择前后的模型性能对比")
            
            st.pyplot(fig)
    
    # 提供参考代码
    with st.expander("查看参考代码"):
        st.code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

# 加载数据集
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 计算基于树的特征重要性
tree_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("基于树的特征重要性:")
print(tree_importances.head(10))

# 计算基于排列的特征重要性
perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)

perm_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': perm_importance.importances_mean
}).sort_values('importance', ascending=False)

print("\\n基于排列的特征重要性:")
print(perm_importances.head(10))

# 可视化两种特征重要性
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(x='importance', y='feature', data=tree_importances.head(10), ax=axes[0])
axes[0].set_title('基于树的特征重要性 (前10个特征)')

sns.barplot(x='importance', y='feature', data=perm_importances.head(10), ax=axes[1])
axes[1].set_title('基于排列的特征重要性 (前10个特征)')

plt.tight_layout()
plt.show()

# 基于特征重要性的特征选择
# 计算累计重要性
tree_importances['cumulative_importance'] = tree_importances['importance'].cumsum() / tree_importances['importance'].sum()

# 设置一个阈值，例如选择累计重要性达到90%的特征
threshold = 0.90
selected_features = tree_importances[tree_importances['cumulative_importance'] <= threshold]['feature'].tolist()

print(f"\\n选择了 {len(selected_features)}/{len(X.columns)} 个特征，累计重要性阈值: {threshold}")
print("选中的特征:", selected_features)

# 使用选定的特征训练新模型
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selected.fit(X_train_selected, y_train)

# 评估原始模型和特征选择后的模型
y_pred_original = rf.predict(X_test)
y_pred_selected = rf_selected.predict(X_test_selected)

accuracy_original = accuracy_score(y_test, y_pred_original)
accuracy_selected = accuracy_score(y_test, y_pred_selected)

print(f"\\n原始模型准确率 (所有特征): {accuracy_original:.4f}")
print(f"特征选择后的模型准确率 ({len(selected_features)} 个特征): {accuracy_selected:.4f}")

# 可视化累计重要性曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(tree_importances) + 1), tree_importances['cumulative_importance'], 'o-')
plt.axhline(y=threshold, color='r', linestyle='--', label=f'{threshold} 阈值')
plt.xlabel('特征数量')
plt.ylabel('累计重要性')
plt.title('特征累计重要性')
plt.legend()
plt.grid(True)
plt.show()
        """, language="python")
    
    st.markdown("""
    ### 思考问题
    
    1. 基于树的特征重要性和基于排列的特征重要性有什么区别？在什么情况下应该选择哪种方法？
    2. 特征选择如何影响模型的性能和复杂度？
    3. 在实际应用中，如何平衡特征重要性、模型复杂度和计算资源的关系？
    """) 