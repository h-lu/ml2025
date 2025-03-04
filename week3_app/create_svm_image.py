import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import os

# 创建目录
os.makedirs('img/week3', exist_ok=True)

# 创建一个线性可分的二分类数据集
X, y = make_classification(
    n_samples=100, n_features=2, n_redundant=0, n_informative=2,
    random_state=42, n_clusters_per_class=1, class_sep=1.0
)

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 训练SVM模型
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_scaled, y)

# 获取超平面参数
w = svm.coef_[0]
b = svm.intercept_[0]
slope = -w[0] / w[1]
xx = np.linspace(-3, 3, 100)
yy = slope * xx - (b / w[1])

# 计算支持向量
margin = 1 / np.sqrt(np.sum(w ** 2))
yy_neg = yy - np.sqrt(1 + slope ** 2) * margin
yy_pos = yy + np.sqrt(1 + slope ** 2) * margin

# 绘制图像
plt.figure(figsize=(10, 8))

# 绘制数据点
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=100, alpha=0.7)

# 绘制决策边界
plt.plot(xx, yy, 'k-', linewidth=2, label='Decision Boundary')
plt.plot(xx, yy_neg, 'k--', linewidth=1, label='Margin')
plt.plot(xx, yy_pos, 'k--', linewidth=1)

# 突出显示支持向量
support_vectors = X_scaled[svm.support_]
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=200, 
            facecolors='none', edgecolors='black', linewidths=2, label='Support Vectors')

# 添加图注
plt.xlabel('Feature 1', fontsize=14)
plt.ylabel('Feature 2', fontsize=14)
plt.title('SVM Decision Boundary and Support Vectors', fontsize=16)
plt.legend(fontsize=12)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.grid(True, linestyle='--', alpha=0.7)

# 保存图像
plt.savefig('img/week3/svm_hyperplane.png', dpi=300, bbox_inches='tight')
plt.close()

print("SVM hyperplane image saved to img/week3/svm_hyperplane.png") 