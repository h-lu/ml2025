import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

def plot_clusters(X, labels, centroids=None, title="聚类结果"):
    """
    绘制聚类结果散点图
    
    参数:
        X: 数据集 shape=(n_samples, 2)
        labels: 簇标签 shape=(n_samples,)
        centroids: 簇中心 shape=(n_clusters, 2)，可选
        title: 图表标题
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
               alpha=0.8, s=50, edgecolors='w')
    
    # 添加图例
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper right", title="簇")
    ax.add_artist(legend1)
    
    # 绘制质心
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], 
                   s=200, marker='X', c='red', 
                   edgecolors='k', label='质心')
        ax.legend()
    
    ax.set_title(title)
    ax.set_xlabel('特征 1')
    ax.set_ylabel('特征 2')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 调整坐标轴，确保比例相同
    ax.set_aspect('equal')
    
    return fig

def plot_kmeans_steps(X, initial_centroids, iterations=3):
    """
    绘制K-means聚类的迭代步骤
    
    参数:
        X: 数据集 shape=(n_samples, 2)
        initial_centroids: 初始中心点 shape=(n_clusters, 2)
        iterations: 要显示的迭代次数
    """
    from sklearn.metrics import pairwise_distances_argmin
    
    n_clusters = initial_centroids.shape[0]
    centroids = initial_centroids.copy()
    
    figs = []
    
    # 绘制初始状态
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X[:, 0], X[:, 1], s=50, alpha=0.8, edgecolors='w')
    ax.scatter(centroids[:, 0], centroids[:, 1], 
              s=200, marker='X', c=['r', 'g', 'b'][:n_clusters], 
              edgecolors='k', label='初始质心')
    ax.set_title('初始状态')
    ax.set_xlabel('特征 1')
    ax.set_ylabel('特征 2')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    figs.append(fig)
    
    colors = ['r', 'g', 'b'][:n_clusters]
    
    for i in range(iterations):
        # 分配步骤
        labels = pairwise_distances_argmin(X, centroids)
        
        # 绘制分配后的结果
        fig, ax = plt.subplots(figsize=(10, 6))
        for k in range(n_clusters):
            cluster_points = X[labels == k]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                      c=colors[k], s=50, alpha=0.8, edgecolors='w')
        ax.scatter(centroids[:, 0], centroids[:, 1], 
                  s=200, marker='X', c=colors, 
                  edgecolors='k', label='质心')
        ax.set_title(f'迭代 {i+1}: 分配阶段')
        ax.set_xlabel('特征 1')
        ax.set_ylabel('特征 2')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        figs.append(fig)
        
        # 更新步骤
        old_centroids = centroids.copy()
        for k in range(n_clusters):
            if np.sum(labels == k) > 0:  # 确保簇非空
                centroids[k] = X[labels == k].mean(axis=0)
        
        # 绘制更新后的结果
        fig, ax = plt.subplots(figsize=(10, 6))
        for k in range(n_clusters):
            cluster_points = X[labels == k]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                      c=colors[k], s=50, alpha=0.8, edgecolors='w')
            
            # 绘制质心移动箭头
            if not np.array_equal(old_centroids[k], centroids[k]):
                ax.arrow(old_centroids[k, 0], old_centroids[k, 1],
                        centroids[k, 0] - old_centroids[k, 0],
                        centroids[k, 1] - old_centroids[k, 1],
                        head_width=0.2, head_length=0.3, fc=colors[k], ec=colors[k])
        
        ax.scatter(centroids[:, 0], centroids[:, 1], 
                  s=200, marker='X', c=colors, 
                  edgecolors='k', label='新质心')
        ax.set_title(f'迭代 {i+1}: 更新阶段')
        ax.set_xlabel('特征 1')
        ax.set_ylabel('特征 2')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        figs.append(fig)
    
    return figs

def plot_elbow_method(X, max_k=10):
    """
    绘制肘部法则图
    
    参数:
        X: 数据集 shape=(n_samples, 2)
        max_k: 最大的簇数
    """
    from sklearn.cluster import KMeans
    
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, max_k + 1), wcss, marker='o', linestyle='-', color='b')
    ax.set_title('肘部法则')
    ax.set_xlabel('簇的数量 (K)')
    ax.set_ylabel('簇内平方和 (WCSS)')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig

def plot_dendrogram(Z, labels=None, title="层次聚类树状图"):
    """
    绘制层次聚类的树状图
    
    参数:
        Z: 层次聚类的 linkage 矩阵
        labels: 叶节点的标签
        title: 图表标题
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    dendrogram(
        Z,
        ax=ax,
        orientation='top',
        labels=labels,
        distance_sort='descending',
        leaf_font_size=10,
        show_leaf_counts=True,
    )
    
    plt.title(title)
    plt.xlabel('样本索引或簇编号')
    plt.ylabel('距离')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig

def plot_silhouette(X, labels, title="轮廓分析"):
    """
    绘制轮廓系数分析图
    
    参数:
        X: 数据集 shape=(n_samples, 2)
        labels: 簇标签 shape=(n_samples,)
        title: 图表标题
    """
    n_clusters = len(np.unique(labels))
    silhouette_avg = silhouette_score(X, labels)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 轮廓系数图
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    
    # 计算每个样本的轮廓系数
    sample_silhouette_values = silhouette_samples(X, labels)
    
    y_lower = 10
    for i in range(n_clusters):
        # 提取当前簇的轮廓系数值
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        
        # 标记簇标签
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        
        # 计算下一个簇的 y_lower
        y_lower = y_upper + 10
    
    ax1.set_title(f"轮廓系数分析 (平均: {silhouette_avg:.2f})")
    ax1.set_xlabel("轮廓系数值")
    ax1.set_ylabel("簇标签")
    
    # 添加平均轮廓系数的竖线
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
    # 绘制聚类结果
    colors = plt.cm.nipy_spectral(labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=50, lw=0, alpha=0.7,
                c=colors, edgecolor='k')
    
    ax2.set_title("聚类结果")
    ax2.set_xlabel("特征 1")
    ax2.set_ylabel("特征 2")
    
    plt.tight_layout()
    
    return fig

def plot_kmeans_centroid_sensitivity(X, K=3, n_init=3, random_state_base=42):
    """
    绘制K-means对初始质心敏感性的图
    
    参数:
        X: 数据集 shape=(n_samples, 2)
        K: 簇数量
        n_init: 不同初始化的数量
        random_state_base: 随机种子基数
    """
    from sklearn.cluster import KMeans
    
    fig, axes = plt.subplots(1, n_init, figsize=(15, 5))
    
    for i in range(n_init):
        random_state = random_state_base + i
        kmeans = KMeans(n_clusters=K, random_state=random_state, n_init=1)
        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_
        
        axes[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                      alpha=0.8, s=50, edgecolors='w')
        axes[i].scatter(centroids[:, 0], centroids[:, 1], 
                      s=200, marker='X', c='red', 
                      edgecolors='k')
        axes[i].set_title(f'初始化 {i+1}')
        axes[i].set_aspect('equal')
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.suptitle("K-means 对初始质心的敏感性", y=1.05)
    
    return fig

def plot_different_linkages(X, linkages=['single', 'complete', 'average', 'ward']):
    """
    绘制不同 linkage 方法的树状图比较
    
    参数:
        X: 数据集 shape=(n_samples, 2)
        linkages: 要比较的 linkage 方法列表
    """
    from scipy.cluster.hierarchy import linkage
    
    n_methods = len(linkages)
    fig, axes = plt.subplots(n_methods, 1, figsize=(12, 4 * n_methods))
    
    for i, method in enumerate(linkages):
        Z = linkage(X, method=method)
        
        if n_methods == 1:
            dendrogram(
                Z,
                ax=axes,
                orientation='top',
                distance_sort='descending',
                leaf_font_size=10,
                show_leaf_counts=True,
            )
            axes.set_title(f"{method.capitalize()} Linkage")
            axes.grid(True, linestyle='--', alpha=0.7)
        else:
            dendrogram(
                Z,
                ax=axes[i],
                orientation='top',
                distance_sort='descending',
                leaf_font_size=10,
                show_leaf_counts=True,
            )
            axes[i].set_title(f"{method.capitalize()} Linkage")
            axes[i].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return fig 