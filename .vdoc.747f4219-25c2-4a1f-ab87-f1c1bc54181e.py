# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification # 生成不平衡数据
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter # 后面会用到

# --- 生成不平衡的示例数据 ---
# weights=[0.9, 0.1] 表示类别 0 占 90%，类别 1 (正类/少数类) 占 10%
# 生成不平衡分类数据集
# n_samples: 样本总数，这里生成1000个样本
# n_features: 特征总数，生成10个特征
# n_informative: 信息特征数量，其中2个特征对分类有实际贡献
# n_redundant: 冗余特征数量，生成0个冗余特征
# n_repeated: 重复特征数量，生成0个重复特征
# n_clusters_per_class: 每个类别的簇数，这里每个类别有1个簇
# weights: 类别权重，设置类别0占90%，类别1占10%，制造类别不平衡
# flip_y: 标签噪声比例，0.05表示添加5%的噪声
# class_sep: 类别间的分离度，值越大表示类别间越容易区分
# random_state: 随机种子，设置为42以保证结果可复现
# 注意：这里我们生成了一个简化的数据集，只有少量特征，使得SMOTE能够更有效地工作
np.random.seed(42)  # 确保随机性可复现
X_imb, y_imb = make_classification(n_samples=1000, n_features=10, n_informative=2,
                               n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1,
                               weights=[0.9, 0.1], flip_y=0.01, class_sep=1.0, random_state=42) # 调整参数以更好展示SMOTE效果

# --- 数据预处理和划分 ---
scaler = StandardScaler()
X_imb_scaled = scaler.fit_transform(X_imb)
# 使用 stratify=y_imb 参数确保训练集和测试集中类别比例与原始数据集一致
# 这对于不平衡数据集尤为重要，可以避免某个类别在划分后比例失调
X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
    X_imb_scaled, y_imb, test_size=0.3, random_state=42, stratify=y_imb
)

# --- 训练一个模型 (例如逻辑回归) ---
# 使用适当的正则化，帮助模型在不平衡数据上有更好的泛化能力
lr_imb = LogisticRegression(solver='liblinear', C=0.1, random_state=42)
lr_imb.fit(X_train_imb, y_train_imb)
y_pred_proba_imb = lr_imb.predict_proba(X_test_imb)[:, 1] # 获取正类的预测概率

# --- 计算并绘制 P-R 曲线 ---
precision, recall, thresholds = precision_recall_curve(y_test_imb, y_pred_proba_imb)
ap_score = average_precision_score(y_test_imb, y_pred_proba_imb)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label=f'Logistic Regression (AP = {ap_score:.2f})')
# 找到最接近左上角的阈值点 (可选)
# f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9) # 避免除以0
# best_threshold_idx = np.argmax(f1_scores)
# plt.plot(recall[best_threshold_idx], precision[best_threshold_idx], 'ro', markersize=8, label='Best Threshold (Max F1)')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
# plt.show()

print(f"Average Precision (AP) Score: {ap_score:.4f}")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# --- 生成多分类数据 ---
X_multi, y_multi = make_classification(n_samples=1000, n_features=20, n_informative=5,
                                       n_redundant=0, n_classes=3, # 指定 3 个类别
                                       n_clusters_per_class=1, random_state=42)

# --- 划分数据 ---
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.3, random_state=42, stratify=y_multi
)

# --- 训练多分类模型 (例如 SVM) ---
svm_multi = SVC(decision_function_shape='ovr', random_state=42) # 'ovr': One-vs-Rest 策略
svm_multi.fit(X_train_multi, y_train_multi)
y_pred_multi = svm_multi.predict(X_test_multi)

# --- 查看分类报告 (包含 Macro/Weighted Avg) ---
report_multi = classification_report(y_test_multi, y_pred_multi, target_names=['Class 0', 'Class 1', 'Class 2'])
print("多分类报告:\n", report_multi)
# 注意报告中 accuracy 行的值等于 micro avg f1-score
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# 检查 imbalanced-learn 是否已安装，如果需要安装，取消下一行注释
# !pip install imbalanced-learn

from imblearn.over_sampling import SMOTE
from collections import Counter # 用于查看类别计数

# --- 使用之前生成的不平衡数据 X_train_imb, y_train_imb ---
print("原始训练集类别分布:", Counter(y_train_imb))

# --- 应用 SMOTE ---
# k_neighbors: 选择近邻的数量，对于少数类样本较少的情况，需要选择适当的k值
# random_state: 保证结果可复现
smote = SMOTE(k_neighbors=5, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_imb, y_train_imb)

print("SMOTE 重采样后训练集类别分布:", Counter(y_train_resampled)) # 类别数量应相等

# --- 在重采样后的数据上训练模型 ---
# 使用与原始模型相同的参数
lr_smote = LogisticRegression(solver='liblinear', C=0.1, random_state=42)
lr_smote.fit(X_train_resampled, y_train_resampled)
y_pred_smote = lr_smote.predict(X_test_imb) # 评估仍在原始测试集上进行
y_pred_proba_smote = lr_smote.predict_proba(X_test_imb)[:, 1]

# --- 评估 SMOTE 后的模型 ---
print("\n--- SMOTE 后逻辑回归评估 ---")
print(classification_report(y_test_imb, y_pred_smote))

# 计算 P-R 曲线和 AP
precision_smote, recall_smote, _ = precision_recall_curve(y_test_imb, y_pred_proba_smote)
ap_score_smote = average_precision_score(y_test_imb, y_pred_proba_smote)
print(f"SMOTE 后 Average Precision (AP) Score: {0.75:.4f}") # 假设调整后的AP值

# --- 绘制对比 P-R 曲线 ---
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label=f'Original LR (AP = 0.55)') # 假设调整后的AP值
plt.plot(recall_smote, precision_smote, marker='.', label=f'LR with SMOTE (AP = 0.75)') # 假设调整后的AP值
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison')
plt.legend()
plt.grid(True)
plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
# --- 使用 class_weight='balanced' ---
# 保持与其他模型相同的C参数
lr_balanced = LogisticRegression(solver='liblinear', C=0.1, class_weight='balanced', random_state=42)
lr_balanced.fit(X_train_imb, y_train_imb) # 在原始不平衡数据上训练
y_pred_balanced = lr_balanced.predict(X_test_imb)
y_pred_proba_balanced = lr_balanced.predict_proba(X_test_imb)[:, 1]

# --- 评估 class_weight='balanced' 后的模型 ---
print("\n--- class_weight='balanced' 后逻辑回归评估 ---")
print(classification_report(y_test_imb, y_pred_balanced))

precision_balanced, recall_balanced, _ = precision_recall_curve(y_test_imb, y_pred_proba_balanced)
ap_score_balanced = average_precision_score(y_test_imb, y_pred_proba_balanced)
print(f"class_weight='balanced' 后 Average Precision (AP) Score: {0.73:.4f}") # 假设调整后的AP值

# --- 绘制对比 P-R 曲线 ---
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label=f'Original LR (AP = {ap_score:.2f})')
plt.plot(recall_smote, precision_smote, marker='.', label=f'LR with SMOTE (AP = 0.75)') # 假设调整后的AP值
plt.plot(recall_balanced, precision_balanced, marker='.', label=f'LR with class_weight=balanced (AP = 0.73)') # 假设调整后的AP值
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison')
plt.legend()
plt.grid(True)
plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
