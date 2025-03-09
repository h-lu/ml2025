import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

from utils.data_loader import load_dataset
from utils.helpers import generate_ecommerce_data

def show():
    st.markdown('<h2 class="sub-header">基础练习</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    本节提供一些基础练习，帮助你巩固对决策树和随机森林算法的理解。每个练习都包含详细的说明和参考代码，你可以在自己的环境中尝试实现。
    """)
    
    # 创建选项卡
    tabs = st.tabs(["练习1: 决策树参数探索", "练习2: 随机森林特征重要性", "练习3: 模型比较", "练习4: 交叉验证与网格搜索"])
    
    # 练习1: 决策树参数探索
    with tabs[0]:
        st.markdown("""
        ## 练习1: 决策树参数探索
        
        ### 目标
        
        探索决策树的不同参数（如最大深度、分裂准则、最小样本数等）对模型性能的影响。
        
        ### 任务
        
        1. 加载鸢尾花数据集（或其他分类数据集）
        2. 划分训练集和测试集
        3. 尝试不同的决策树参数，训练多个模型
        4. 比较不同参数下模型的性能
        5. 可视化决策树结构，观察参数变化对树结构的影响
        
        ### 参考代码
        """)
        
        st.code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# 加载数据集
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 探索不同的最大深度
max_depths = [1, 2, 3, 5, 10, None]
train_scores = []
test_scores = []

for depth in max_depths:
    # 创建决策树模型
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    
    # 训练模型
    dt.fit(X_train, y_train)
    
    # 预测
    y_train_pred = dt.predict(X_train)
    y_test_pred = dt.predict(X_test)
    
    # 计算准确率
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    train_scores.append(train_acc)
    test_scores.append(test_acc)
    
    print(f"Max Depth: {depth}")
    print(f"训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    print("-" * 50)

# 可视化结果
plt.figure(figsize=(10, 6))
plt.plot(max_depths, train_scores, 'o-', label='训练集准确率')
plt.plot(max_depths, test_scores, 'o-', label='测试集准确率')
plt.xlabel('最大深度')
plt.ylabel('准确率')
plt.title('决策树最大深度与准确率的关系')
plt.legend()
plt.grid(True)
plt.show()

# 可视化决策树 (选择一个合适的深度)
plt.figure(figsize=(15, 10))
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
plot_tree(dt, filled=True, feature_names=feature_names, class_names=target_names)
plt.title('决策树可视化 (max_depth=3)')
plt.show()
        """, language="python")
        
        st.markdown("""
        ### 思考问题
        
        1. 最大深度如何影响模型的过拟合和欠拟合？
        2. 分裂准则（gini vs entropy）对模型性能有显著影响吗？
        3. 如何选择最优的决策树参数？
        """)
    
    # 练习2: 随机森林特征重要性
    with tabs[1]:
        st.markdown("""
        ## 练习2: 随机森林特征重要性
        
        ### 目标
        
        使用随机森林算法分析特征重要性，了解哪些特征对预测结果影响最大。
        
        ### 任务
        
        1. 加载电商用户行为数据集（或其他多特征分类数据集）
        2. 划分训练集和测试集
        3. 训练随机森林模型
        4. 提取并可视化特征重要性
        5. 基于特征重要性进行特征选择，重新训练模型并比较性能
        
        ### 参考代码
        """)
        
        st.code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 生成模拟电商用户行为数据
# 实际应用中，你可以加载自己的数据集
np.random.seed(42)
n_samples = 1000

# 用户基本信息
age = np.random.normal(35, 10, n_samples).astype(int)
age = np.clip(age, 18, 70)  # 限制年龄范围
gender = np.random.binomial(1, 0.5, n_samples)  # 0=女性, 1=男性

# 用户行为数据
visit_frequency = np.random.poisson(10, n_samples)  # 访问频率
avg_time_spent = np.random.gamma(5, 2, n_samples)  # 平均停留时间(分钟)
items_viewed = np.random.poisson(15, n_samples)  # 浏览商品数
items_purchased = np.random.binomial(items_viewed, 0.3, n_samples)  # 购买商品数

# 购物车相关
cart_abandonment = np.random.binomial(1, 0.4, n_samples)  # 购物车放弃率
cart_value = np.random.gamma(100, 10, n_samples)  # 购物车价值

# 用户历史
purchase_history = np.random.gamma(500, 20, n_samples)  # 历史购买金额
return_rate = np.random.beta(2, 10, n_samples)  # 退货率

# 生成目标变量: 高价值(1)vs低价值(0)用户
# 基于用户行为指标计算一个用户价值分数
value_score = (
    visit_frequency * 0.2 + 
    items_purchased * 0.3 + 
    purchase_history * 0.002 - 
    cart_abandonment * 5 - 
    return_rate * 10
)

# 将分数转换为二元目标
threshold = np.percentile(value_score, 70)  # 设置为前30%为高价值用户
user_value = (value_score > threshold).astype(int)

# 创建DataFrame
data = {
    'age': age,
    'gender': gender,
    'visit_frequency': visit_frequency,
    'avg_time_spent': avg_time_spent,
    'items_viewed': items_viewed,
    'items_purchased': items_purchased,
    'cart_abandonment': cart_abandonment,
    'cart_value': cart_value,
    'purchase_history': purchase_history,
    'return_rate': return_rate,
    'high_value_user': user_value
}

df = pd.DataFrame(data)

# 划分特征和目标变量
X = df.drop('high_value_user', axis=1)
y = df['high_value_user']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 预测
y_pred = rf.predict(X_test)
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print("\\n分类报告:\\n", classification_report(y_test, y_pred))

# 提取特征重要性
feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\\n特征重要性:\\n", feature_importances)

# 可视化特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title('随机森林特征重要性')
plt.tight_layout()
plt.show()

# 基于特征重要性进行特征选择
# 选择重要性排名前5的特征
top_features = feature_importances['feature'].head(5).tolist()
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

# 使用选定的特征重新训练模型
rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selected.fit(X_train_selected, y_train)

# 预测
y_pred_selected = rf_selected.predict(X_test_selected)
print(f"\\n使用前5个重要特征的准确率: {accuracy_score(y_test, y_pred_selected):.4f}")
print("\\n分类报告:\\n", classification_report(y_test, y_pred_selected))
        """, language="python")
        
        st.markdown("""
        ### 思考问题
        
        1. 哪些特征对预测高价值用户最重要？这与你的业务理解是否一致？
        2. 使用所有特征和只使用重要特征的模型性能有何不同？
        3. 特征重要性如何帮助我们理解模型的决策过程？
        """)
    
    # 练习3: 模型比较
    with tabs[2]:
        st.markdown("""
        ## 练习3: 模型比较
        
        ### 目标
        
        比较决策树和随机森林在同一数据集上的性能差异。
        
        ### 任务
        
        1. 加载分类数据集
        2. 划分训练集和测试集
        3. 分别训练决策树和随机森林模型
        4. 比较两种模型的性能指标（准确率、精确率、召回率、F1-score等）
        5. 分析两种模型的优缺点
        
        ### 参考代码
        """)
        
        st.code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# 加载数据集
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
target_names = data.target_names

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练决策树模型
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:, 1]

# 训练随机森林模型
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# 计算性能指标
models = ['决策树', '随机森林']
accuracy = [accuracy_score(y_test, y_pred_dt), accuracy_score(y_test, y_pred_rf)]
precision = [precision_score(y_test, y_pred_dt), precision_score(y_test, y_pred_rf)]
recall = [recall_score(y_test, y_pred_dt), recall_score(y_test, y_pred_rf)]
f1 = [f1_score(y_test, y_pred_dt), f1_score(y_test, y_pred_rf)]

# 创建性能指标DataFrame
performance = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-score': f1
})

print("模型性能比较:")
print(performance)

# 可视化性能指标
plt.figure(figsize=(12, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
bar_width = 0.35
index = np.arange(len(metrics))

plt.bar(index, performance.iloc[0, 1:].values, bar_width, label='决策树')
plt.bar(index + bar_width, performance.iloc[1, 1:].values, bar_width, label='随机森林')

plt.xlabel('评估指标')
plt.ylabel('得分')
plt.title('决策树 vs 随机森林性能比较')
plt.xticks(index + bar_width / 2, metrics)
plt.legend()
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# 绘制ROC曲线
plt.figure(figsize=(8, 6))

# 决策树ROC曲线
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)
plt.plot(fpr_dt, tpr_dt, label=f'决策树 (AUC = {roc_auc_dt:.3f})')

# 随机森林ROC曲线
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.plot(fpr_rf, tpr_rf, label=f'随机森林 (AUC = {roc_auc_rf:.3f})')

# 随机猜测的基准线
plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')

plt.xlabel('假正例率 (FPR)')
plt.ylabel('真正例率 (TPR)')
plt.title('ROC曲线')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
        """, language="python")
        
        st.markdown("""
        ### 思考问题
        
        1. 随机森林相比决策树在哪些方面表现更好？为什么？
        2. 两种模型的ROC曲线有何不同？这说明了什么？
        3. 在实际应用中，你会选择哪种模型？为什么？
        """)
    
    # 练习4: 交叉验证与网格搜索
    with tabs[3]:
        st.markdown("""
        ## 练习4: 交叉验证与网格搜索
        
        ### 目标
        
        使用交叉验证和网格搜索找到随机森林的最优参数。
        
        ### 任务
        
        1. 加载分类数据集
        2. 使用K折交叉验证评估随机森林模型
        3. 定义参数网格，使用网格搜索找到最优参数
        4. 使用最优参数训练最终模型并评估性能
        
        ### 参考代码
        """)
        
        st.code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 加载数据集
data = load_wine()
X, y = data.data, data.target
feature_names = data.feature_names
target_names = data.target_names

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建基础随机森林模型
rf = RandomForestClassifier(random_state=42)

# 使用5折交叉验证评估模型
cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
print(f"交叉验证得分: {cv_scores}")
print(f"平均得分: {cv_scores.mean():.4f}, 标准差: {cv_scores.std():.4f}")

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 创建网格搜索对象
grid_search = GridSearchCV(
    rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)

# 执行网格搜索
grid_search.fit(X_train, y_train)

# 输出最佳参数和得分
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")

# 使用最佳参数的模型进行预测
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

# 评估最终模型
print(f"测试集准确率: {accuracy_score(y_test, y_pred):.4f}")
print("\\n分类报告:\\n", classification_report(y_test, y_pred, target_names=target_names))

# 可视化网格搜索结果 (n_estimators参数)
plt.figure(figsize=(10, 6))
n_estimators_results = []
for params, mean_score, std_score in zip(grid_search.cv_results_['params'], 
                                         grid_search.cv_results_['mean_test_score'], 
                                         grid_search.cv_results_['std_test_score']):
    if params['max_depth'] == grid_search.best_params_['max_depth'] and \\
       params['min_samples_split'] == grid_search.best_params_['min_samples_split'] and \\
       params['min_samples_leaf'] == grid_search.best_params_['min_samples_leaf']:
        n_estimators_results.append((params['n_estimators'], mean_score, std_score))

n_estimators_results.sort(key=lambda x: x[0])
n_estimators = [x[0] for x in n_estimators_results]
mean_scores = [x[1] for x in n_estimators_results]
std_scores = [x[2] for x in n_estimators_results]

plt.errorbar(n_estimators, mean_scores, yerr=std_scores, marker='o')
plt.xlabel('n_estimators')
plt.ylabel('交叉验证得分')
plt.title('n_estimators参数对模型性能的影响')
plt.grid(True)
plt.show()
        """, language="python")
        
        st.markdown("""
        ### 思考问题
        
        1. 交叉验证如何帮助我们更可靠地评估模型性能？
        2. 网格搜索找到的最优参数是否总是最好的？为什么？
        3. 如何平衡模型性能和计算开销？
        """)
    
    # 提供一个简单的交互式示例
    st.markdown("## 交互式示例：电商用户分类")
    
    # 生成示例数据
    if st.button("生成示例数据"):
        df = generate_ecommerce_data(n_samples=1000, random_state=42)
        st.dataframe(df.head())
        
        # 划分特征和目标变量
        X = df.drop(['user_id', 'high_value_user'], axis=1)
        y = df['high_value_user']
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # 训练决策树和随机森林模型
        dt = DecisionTreeClassifier(max_depth=5, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        
        dt.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        
        # 预测
        y_pred_dt = dt.predict(X_test)
        y_pred_rf = rf.predict(X_test)
        
        # 计算准确率
        acc_dt = accuracy_score(y_test, y_pred_dt)
        acc_rf = accuracy_score(y_test, y_pred_rf)
        
        # 显示结果
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 决策树模型")
            st.markdown(f"准确率: {acc_dt:.4f}")
            st.markdown("混淆矩阵:")
            cm_dt = confusion_matrix(y_test, y_pred_dt)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('预测标签')
            ax.set_ylabel('真实标签')
            ax.set_title('决策树混淆矩阵')
            st.pyplot(fig)
        
        with col2:
            st.markdown("### 随机森林模型")
            st.markdown(f"准确率: {acc_rf:.4f}")
            st.markdown("混淆矩阵:")
            cm_rf = confusion_matrix(y_test, y_pred_rf)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('预测标签')
            ax.set_ylabel('真实标签')
            ax.set_title('随机森林混淆矩阵')
            st.pyplot(fig)
        
        # 特征重要性
        st.markdown("### 随机森林特征重要性")
        feature_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importances, ax=ax)
        ax.set_title('随机森林特征重要性')
        st.pyplot(fig) 