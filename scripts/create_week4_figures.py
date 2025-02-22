import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS
plt.rcParams['axes.unicode_minus'] = False

def plot_decision_tree():
    """创建一个简单直观的决策树示例"""
    # 创建简单的示例数据
    # 特征：年龄、消费金额
    X = np.array([
        [25, 2000], [35, 3000], [45, 2500], [20, 1500], [30, 4000],
        [50, 5000], [40, 3500], [22, 1800], [38, 3200], [42, 4500]
    ])
    # 标签：是否高价值客户
    y = np.array([0, 1, 0, 0, 1, 1, 1, 0, 1, 1])
    
    # 训练决策树
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    
    # 绘制决策树
    plt.figure(figsize=(15, 10))
    plot_tree(clf, 
             feature_names=['年龄', '消费金额'],
             class_names=['普通客户', '高价值客户'],
             filled=True, rounded=True, fontsize=10)
    plt.title('电商用户价值决策树示例', fontsize=12, pad=20)
    plt.savefig('img/week4/decision_tree.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_entropy_gini():
    """创建信息熵和基尼系数对比图"""
    p = np.linspace(0, 1, 100)
    entropy = -p * np.log2(p + 1e-10) - (1-p) * np.log2(1-p + 1e-10)
    gini = 1 - p**2 - (1-p)**2
    
    plt.figure(figsize=(10, 6))
    plt.plot(p, entropy, label='信息熵', linewidth=2, color='#2ecc71')
    plt.plot(p, gini, label='基尼系数', linewidth=2, color='#e74c3c')
    plt.xlabel('正类概率 p')
    plt.ylabel('不纯度')
    plt.title('信息熵与基尼系数对比')
    plt.legend(loc='center right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('img/week4/entropy_gini.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_bagging():
    """创建Bagging方法示意图"""
    plt.figure(figsize=(12, 8))
    
    # 定义布局网格
    gs = plt.GridSpec(3, 3)
    
    # 原始数据集
    ax1 = plt.subplot(gs[0, 1])
    plt.text(0.5, 0.5, '原始数据集\nN个样本', ha='center', va='center', fontsize=12)
    plt.axis('off')
    
    # 自助采样数据集
    for i in range(3):
        ax = plt.subplot(gs[1, i])
        plt.text(0.5, 0.5, f'自助采样数据集 {i+1}\n(N个样本)', ha='center', va='center', fontsize=12)
        plt.axis('off')
    
    # 基学习器和最终结果
    for i in range(3):
        ax = plt.subplot(gs[2, i])
        if i < 2:
            plt.text(0.5, 0.5, f'基学习器 {i+1}\n(决策树)', ha='center', va='center', fontsize=12)
        else:
            plt.text(0.5, 0.5, '集成结果\n(投票/平均)', ha='center', va='center', fontsize=12)
        plt.axis('off')
    
    # 添加箭头连接
    plt.annotate('', xy=(0.3, 0.65), xytext=(0.5, 0.85),
                 xycoords='figure fraction', 
                 arrowprops=dict(arrowstyle='->', color='#3498db', lw=2))
    plt.annotate('', xy=(0.5, 0.65), xytext=(0.5, 0.85),
                 xycoords='figure fraction', 
                 arrowprops=dict(arrowstyle='->', color='#3498db', lw=2))
    plt.annotate('', xy=(0.7, 0.65), xytext=(0.5, 0.85),
                 xycoords='figure fraction', 
                 arrowprops=dict(arrowstyle='->', color='#3498db', lw=2))
    
    # 添加最终集成的箭头
    plt.annotate('', xy=(0.8, 0.25), xytext=(0.5, 0.4),
                 xycoords='figure fraction', 
                 arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))
    
    plt.suptitle('Bagging方法示意图', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig('img/week4/bagging.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_random_forest_feature_selection():
    """创建随机森林特征随机选择示意图"""
    plt.figure(figsize=(12, 8))
    
    # 所有特征
    features = ['购买频率', '浏览时长', '购物车数量', '收藏数量', 
                '优惠券使用', '搜索次数', '评论数量', '退货率']
    
    # 创建三个子图
    gs = plt.GridSpec(2, 2)
    
    # 所有特征
    ax1 = plt.subplot(gs[0, :])
    for i, feature in enumerate(features):
        plt.text(i/len(features), 0.5, feature, 
                ha='center', va='center', fontsize=10,
                bbox=dict(facecolor='#3498db', alpha=0.3))
    plt.title('所有特征 (M=8个)')
    plt.axis('off')
    
    # 随机特征子集
    for i in range(2):
        ax = plt.subplot(gs[1, i])
        # 随机选择特征子集
        np.random.seed(i)
        subset = np.random.choice(features, size=3, replace=False)
        for j, feature in enumerate(subset):
            plt.text(j/3, 0.5, feature,
                    ha='center', va='center', fontsize=10,
                    bbox=dict(facecolor='#e74c3c', alpha=0.3))
        plt.title(f'随机特征子集 {i+1} (m=3个)')
        plt.axis('off')
    
    # 添加箭头
    plt.annotate('特征随机选择', xy=(0.3, 0.5), xytext=(0.3, 0.7),
                 xycoords='figure fraction',
                 arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=2))
    plt.annotate('特征随机选择', xy=(0.7, 0.5), xytext=(0.7, 0.7),
                 xycoords='figure fraction',
                 arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=2))
    
    plt.suptitle('随机森林特征随机选择示意图', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig('img/week4/random_forest_feature_selection.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance():
    """创建随机森林特征重要性示例图"""
    # 使用更有意义的特征名称和重要性分数
    feature_names = ['购买频率', '浏览时长', '购物车数量', 
                    '收藏数量', '优惠券使用', '搜索次数',
                    '评论数量', '退货率', '客单价', '会员等级']
    
    importances = np.array([0.25, 0.18, 0.15, 0.12, 0.08, 
                           0.07, 0.06, 0.04, 0.03, 0.02])
    
    # 创建DataFrame并排序
    df = pd.DataFrame({'特征': feature_names, '重要性': importances})
    df = df.sort_values('重要性', ascending=True)
    
    # 创建颜色渐变
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df['特征'], df['重要性'], color=colors)
    
    # 添加数值标签
    for i, v in enumerate(df['重要性']):
        plt.text(v, i, f' {v:.3f}', va='center')
    
    plt.title('随机森林特征重要性分析')
    plt.xlabel('特征重要性得分')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('img/week4/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_cross_validation():
    """创建5折交叉验证示意图"""
    plt.figure(figsize=(12, 8))
    n_splits = 5
    
    # 创建示意图
    for i in range(n_splits):
        plt.subplot(n_splits, 1, i+1)
        
        # 计算验证集的位置
        val_start = i * 0.2
        
        # 训练集（分成两部分，验证集前后）
        if i > 0:
            plt.barh(0, val_start, height=0.3, left=0, 
                    color='#3498db', alpha=0.7, label='训练集')
        if i < n_splits-1:
            plt.barh(0, 1-val_start-0.2, height=0.3, 
                    left=val_start+0.2, color='#3498db', alpha=0.7)
        
        # 验证集（20%的连续数据）
        plt.barh(0, 0.2, height=0.3, left=val_start, 
                color='#e74c3c', alpha=0.7, label='验证集')
        
        if i == 0:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.yticks([])
        plt.xticks([])
        plt.title(f'第 {i+1} 折')
    
    plt.suptitle('5折交叉验证示意图', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig('img/week4/cross_validation.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # 创建图片目录
    import os
    os.makedirs('img/week4', exist_ok=True)
    
    # 生成所有图
    plot_decision_tree()
    plot_entropy_gini()
    plot_bagging()
    plot_random_forest_feature_selection()
    plot_feature_importance()
    plot_cross_validation() 