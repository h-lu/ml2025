#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成用于可视化的SVG文件
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
import platform

# 配置matplotlib支持中文显示，根据操作系统选择不同的字体
system = platform.system()
try:
    if system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
    elif system == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC', 'Hiragino Sans GB', 'STHeiti']
    elif system == 'Linux':
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK JP', 'Noto Sans CJK SC', 'Droid Sans Fallback']
    else:
        # 默认字体设置
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
except Exception as e:
    print(f"无法设置中文字体: {e}。图表中的中文可能无法正确显示。")

# 创建目录
os.makedirs('img/extra', exist_ok=True)

# 创建混淆矩阵SVG文件
X, y = make_classification(random_state=42)
cm = np.array([[85, 15], [10, 90]])  # 简单的混淆矩阵示例

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['预测负例', '预测正例'],
            yticklabels=['实际负例', '实际正例'])
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵示例')
plt.tight_layout()
plt.savefig('img/extra/confusion_matrix.svg', format='svg')
print('混淆矩阵SVG文件已生成: img/extra/confusion_matrix.svg')
plt.close()

# 创建学习曲线SVG文件
train_sizes = np.linspace(0.1, 1.0, 10)
train_mean = [0.65, 0.70, 0.74, 0.76, 0.78, 0.80, 0.82, 0.83, 0.84, 0.85]
train_std = [0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01]
test_mean = [0.60, 0.65, 0.68, 0.70, 0.71, 0.72, 0.72, 0.73, 0.73, 0.73]
test_std = [0.08, 0.07, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03]

plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, np.array(train_mean) - np.array(train_std),
                np.array(train_mean) + np.array(train_std), alpha=0.1, color='r')
plt.fill_between(train_sizes, np.array(test_mean) - np.array(test_std),
                np.array(test_mean) + np.array(test_std), alpha=0.1, color='g')
plt.plot(train_sizes, train_mean, 'o-', color='r', label='训练集得分')
plt.plot(train_sizes, test_mean, 'o-', color='g', label='验证集得分')
plt.xlabel('训练样本数')
plt.ylabel('得分')
plt.title('学习曲线')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig('img/extra/learning_curve.svg', format='svg')
print('学习曲线SVG文件已生成: img/extra/learning_curve.svg')
plt.close()

print('所有SVG文件生成完成！') 