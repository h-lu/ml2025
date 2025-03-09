import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score

from utils.data_loader import load_dataset

def show_ensemble_comparison():
    st.markdown("## 高级练习1: 集成学习方法比较")
    
    st.markdown("""
    ### 目标
    
    比较不同集成学习方法（随机森林、Gradient Boosting、AdaBoost）的性能差异。
    
    ### 任务
    
    1. 加载分类数据集
    2. 划分训练集和测试集
    3. 训练不同的集成学习模型
    4. 比较各模型的性能指标
    5. 分析不同集成学习方法的优缺点
    """)
    
    # 数据集选择
    dataset_name = st.selectbox(
        "选择数据集",
        ["iris", "wine", "breast_cancer", "ecommerce"],
        index=2,
        key="ensemble_comparison_dataset"
    )
    
    # 加载数据集
    X_train, X_test, y_train, y_test, feature_names, target_names = load_dataset(
        dataset_name, test_size=0.2, random_state=42
    )
    
    # 参数设置
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rf_n_estimators = st.slider(
            "随机森林树数量", 
            min_value=10, 
            max_value=200, 
            value=100,
            step=10,
            key="rf_n_estimators"
        )
        
    with col2:
        gb_n_estimators = st.slider(
            "Gradient Boosting树数量", 
            min_value=10, 
            max_value=200, 
            value=100,
            step=10,
            key="gb_n_estimators"
        )
        
    with col3:
        ada_n_estimators = st.slider(
            "AdaBoost树数量", 
            min_value=10, 
            max_value=200, 
            value=100,
            step=10,
            key="ada_n_estimators"
        )
    
    # 创建按钮运行分析
    if st.button("运行集成学习比较分析", key="run_ensemble_comparison"):
        with st.spinner("正在训练和评估模型..."):
            # 创建不同的集成学习模型
            rf_model = RandomForestClassifier(
                n_estimators=rf_n_estimators, 
                random_state=42
            )
            
            gb_model = GradientBoostingClassifier(
                n_estimators=gb_n_estimators, 
                random_state=42
            )
            
            base_estimator = DecisionTreeClassifier(max_depth=1)
            ada_model = AdaBoostClassifier(
                base_estimator=base_estimator,
                n_estimators=ada_n_estimators, 
                random_state=42
            )
            
            # 训练模型
            models = {
                "随机森林 (Random Forest)": rf_model,
                "Gradient Boosting": gb_model,
                "AdaBoost": ada_model
            }
            
            results = []
            
            # 为每个模型计算性能指标
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # 对于二分类问题计算ROC曲线
                if len(np.unique(y_test)) == 2:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                else:
                    fpr, tpr, roc_auc = None, None, None
                
                results.append({
                    "模型": name,
                    "准确率": accuracy_score(y_test, y_pred),
                    "精确率": precision_score(y_test, y_pred, average='weighted'),
                    "召回率": recall_score(y_test, y_pred, average='weighted'),
                    "F1分数": f1_score(y_test, y_pred, average='weighted'),
                    "FPR": fpr,
                    "TPR": tpr,
                    "AUC": roc_auc,
                    "预测": y_pred
                })
            
            # 显示性能指标比较
            performance_df = pd.DataFrame([
                {
                    "模型": r["模型"],
                    "准确率": r["准确率"],
                    "精确率": r["精确率"],
                    "召回率": r["召回率"],
                    "F1分数": r["F1分数"],
                    "AUC": r["AUC"] if r["AUC"] is not None else "N/A"
                }
                for r in results
            ])
            
            st.markdown("### 模型性能比较")
            st.dataframe(performance_df.style.highlight_max(subset=["准确率", "精确率", "召回率", "F1分数"]))
            
            # 绘制性能指标比较图
            fig, ax = plt.subplots(figsize=(10, 6))
            
            metrics = ["准确率", "精确率", "召回率", "F1分数"]
            x = np.arange(len(metrics))
            width = 0.25
            
            for i, result in enumerate(results):
                values = [result["准确率"], result["精确率"], result["召回率"], result["F1分数"]]
                ax.bar(x + i*width, values, width, label=result["模型"])
            
            ax.set_ylabel('分数')
            ax.set_title('不同集成学习方法的性能比较')
            ax.set_xticks(x + width)
            ax.set_xticklabels(metrics)
            ax.legend()
            
            st.pyplot(fig)
            
            # 如果是二分类问题，绘制ROC曲线
            if len(np.unique(y_test)) == 2:
                st.markdown("### ROC曲线比较")
                
                fig, ax = plt.subplots(figsize=(8, 6))
                
                for result in results:
                    if result["FPR"] is not None and result["TPR"] is not None:
                        ax.plot(
                            result["FPR"], 
                            result["TPR"], 
                            label=f'{result["模型"]} (AUC = {result["AUC"]:.3f})'
                        )
                
                # 添加随机猜测的基准线
                ax.plot([0, 1], [0, 1], 'k--', label='随机猜测')
                ax.set_xlabel('假正例率 (FPR)')
                ax.set_ylabel('真正例率 (TPR)')
                ax.set_title('ROC曲线比较')
                ax.legend(loc='lower right')
                ax.grid(True)
                
                st.pyplot(fig)
            
            # 混淆矩阵对比
            st.markdown("### 混淆矩阵对比")
            
            cols = st.columns(len(models))
            
            for i, result in enumerate(results):
                with cols[i]:
                    st.markdown(f"**{result['模型']}**")
                    cm = confusion_matrix(y_test, result["预测"])
                    
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_xlabel("预测标签")
                    ax.set_ylabel("真实标签")
                    ax.set_title(f"{result['模型']} 混淆矩阵")
                    
                    st.pyplot(fig)
    
    # 提供参考代码
    with st.expander("查看参考代码"):
        st.code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# 加载数据集
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
target_names = data.target_names

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建不同的集成学习模型
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100, 
        random_state=42
    )
}

# 评估每个模型
results = {}

for name, model in models.items():
    # 使用交叉验证评估模型
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    # 在训练集上训练模型
    model.fit(X_train, y_train)
    
    # 在测试集上评估模型
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 计算各种性能指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # 计算ROC曲线和AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # 存储结果
    results[name] = {
        "CV Scores": cv_scores,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC": roc_auc,
        "FPR": fpr,
        "TPR": tpr
    }

# 输出结果
for name, result in results.items():
    print(f"\\n{name}:")
    print(f"CV Scores: {result['CV Scores']}")
    print(f"Mean CV Score: {result['CV Scores'].mean():.4f}")
    print(f"Accuracy: {result['Accuracy']:.4f}")
    print(f"Precision: {result['Precision']:.4f}")
    print(f"Recall: {result['Recall']:.4f}")
    print(f"F1 Score: {result['F1 Score']:.4f}")
    print(f"AUC: {result['AUC']:.4f}")

# 绘制ROC曲线
plt.figure(figsize=(10, 8))

for name, result in results.items():
    plt.plot(
        result["FPR"], 
        result["TPR"], 
        label=f'{name} (AUC = {result["AUC"]:.3f})'
    )

# 添加随机猜测的基准线
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# 绘制性能指标比较条形图
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
model_names = list(results.keys())

values = np.zeros((len(model_names), len(metrics)))
for i, name in enumerate(model_names):
    values[i, 0] = results[name]["Accuracy"]
    values[i, 1] = results[name]["Precision"]
    values[i, 2] = results[name]["Recall"]
    values[i, 3] = results[name]["F1 Score"]

x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 8))
for i in range(len(model_names)):
    ax.bar(x + i*width, values[i], width, label=model_names[i])

ax.set_ylabel('Score')
ax.set_title('Performance Metrics Comparison')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.legend()

plt.tight_layout()
plt.show()
        """, language="python")
    
    st.markdown("""
    ### 思考问题
    
    1. 不同集成学习方法在哪些场景下表现更好？
    2. Bagging方法（如随机森林）和Boosting方法（如Gradient Boosting、AdaBoost）的主要区别是什么？
    3. 如何根据数据特点和问题目标选择合适的集成学习方法？
    """) 