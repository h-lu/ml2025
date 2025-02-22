import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS
plt.rcParams['axes.unicode_minus'] = False

def plot_sigmoid():
    """绘制Sigmoid函数图"""
    x = np.linspace(-10, 10, 100)
    y = 1 / (1 + np.exp(-x))
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.grid(True)
    plt.title('Sigmoid函数')
    plt.xlabel('z')
    plt.ylabel('σ(z)')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.3)
    plt.savefig('sigmoid.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_logistic_boundary():
    """绘制逻辑回归决策边界"""
    # 生成示例数据
    X, y = make_blobs(n_samples=100, centers=2, random_state=42)
    
    # 训练逻辑回归模型
    clf = LogisticRegression()
    clf.fit(X, y)
    
    # 创建网格点
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # 预测网格点的类别
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title('逻辑回归决策边界')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.savefig('logistic_boundary.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_svm_margin():
    """绘制SVM最大间隔和支持向量"""
    # 生成线性可分的数据
    X, y = make_blobs(n_samples=50, centers=2, random_state=42)
    
    # 训练SVM模型
    clf = SVC(kernel='linear')
    clf.fit(X, y)
    
    # 创建网格点
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # 预测网格点的类别
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    
    # 绘制支持向量
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
               s=100, linewidth=1, facecolors='none', edgecolors='k',
               label='支持向量')
    
    plt.title('SVM最大间隔和支持向量')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.legend()
    plt.savefig('svm_margin.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_kernel_comparison():
    """绘制不同核函数的效果对比"""
    # 生成月牙形数据
    X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
    
    # 定义不同的核函数
    kernels = ['linear', 'poly', 'rbf']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, kernel in enumerate(kernels):
        # 训练SVM模型
        clf = SVC(kernel=kernel)
        clf.fit(X, y)
        
        # 创建网格点
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        
        # 预测网格点的类别
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # 绘制决策边界
        axes[i].contourf(xx, yy, Z, alpha=0.4)
        axes[i].scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        axes[i].set_title(f'{kernel}核函数')
    
    plt.tight_layout()
    plt.savefig('kernel_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve():
    """绘制ROC曲线示例"""
    # 生成示例数据和预测概率
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_scores = np.random.random(100)
    
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC曲线 (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (False Positive Rate)')
    plt.ylabel('真正例率 (True Positive Rate)')
    plt.title('ROC曲线示例')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # 生成所有图
    plot_sigmoid()
    plot_logistic_boundary()
    plot_svm_margin()
    plot_kernel_comparison()
    plot_roc_curve() 