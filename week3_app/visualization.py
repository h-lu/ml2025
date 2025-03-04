import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def plot_decision_boundary(model, X, y, ax=None, title=None):
    """绘制决策边界"""
    if ax is None:
        ax = plt.gca()
    
    # 创建网格点
    h = 0.02  # 网格步长
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # 预测网格点的类别
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界和样本点
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    
    if title:
        ax.set_title(title)
    
    return ax

def generate_dataset(dataset_type, n_samples=300, noise=0.1, random_state=42):
    """生成不同类型的数据集"""
    if dataset_type == "线性可分":
        X, y = make_classification(
            n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2,
            random_state=random_state, n_clusters_per_class=1, class_sep=1.5
        )
    elif dataset_type == "同心圆":
        X, y = make_circles(
            n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state
        )
    elif dataset_type == "半月形":
        X, y = make_moons(
            n_samples=n_samples, noise=noise, random_state=random_state
        )
    
    return X, y

def show_visualization():
    """显示算法可视化页面"""
    
    st.header("算法可视化")
    
    st.markdown("""
    在本节中，我们将通过交互式可视化来展示逻辑回归和SVM的工作原理和决策边界。
    您可以通过调整各种参数，直观地观察这些算法如何在不同类型的数据集上表现。
    """)
    
    # 创建列布局
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("参数设置")
        
        # 数据集选择
        dataset_type = st.selectbox(
            "选择数据集类型",
            ["线性可分", "同心圆", "半月形"]
        )
        
        # 样本数量
        n_samples = st.slider("样本数量", min_value=50, max_value=500, value=200, step=50)
        
        # 噪声水平
        noise = st.slider("噪声水平", min_value=0.0, max_value=0.5, value=0.1, step=0.05)
        
        # 随机种子
        random_state = st.slider("随机种子", min_value=0, max_value=100, value=42, step=1)
        
        # 算法选择
        algorithm = st.radio("选择算法", ["逻辑回归", "SVM", "两者对比"])
        
        # 算法参数
        if algorithm in ["逻辑回归", "两者对比"]:
            st.markdown("##### 逻辑回归参数")
            lr_C = st.slider("正则化强度C (逻辑回归)", min_value=0.01, max_value=10.0, value=1.0, step=0.1, key="lr_C")
        
        if algorithm in ["SVM", "两者对比"]:
            st.markdown("##### SVM参数")
            svm_C = st.slider("正则化强度C (SVM)", min_value=0.01, max_value=10.0, value=1.0, step=0.1, key="svm_C")
            svm_kernel = st.selectbox("核函数", ["linear", "rbf", "poly"])
            
            if svm_kernel == "rbf":
                svm_gamma = st.slider("Gamma参数", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
            elif svm_kernel == "poly":
                svm_degree = st.slider("多项式次数", min_value=2, max_value=5, value=3, step=1)
        
        generate_btn = st.button("生成可视化")
    
    with col2:
        st.subheader("可视化结果")
        
        if 'generate_btn' not in locals() or generate_btn:
            # 生成数据集
            X, y = generate_dataset(dataset_type, n_samples, noise, random_state)
            
            # 标准化数据
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=random_state)
            
            if algorithm == "逻辑回归":
                # 训练逻辑回归模型
                lr_model = LogisticRegression(C=lr_C, random_state=random_state)
                lr_model.fit(X_train, y_train)
                
                # 计算准确率
                lr_train_acc = accuracy_score(y_train, lr_model.predict(X_train))
                lr_test_acc = accuracy_score(y_test, lr_model.predict(X_test))
                
                # 绘制决策边界
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_decision_boundary(lr_model, X_scaled, y, ax=ax, 
                                      title=f"逻辑回归决策边界 (C={lr_C})\n训练准确率: {lr_train_acc:.2f}, 测试准确率: {lr_test_acc:.2f}")
                st.pyplot(fig)
                
            elif algorithm == "SVM":
                # SVM参数设置
                if svm_kernel == "rbf":
                    svm_model = SVC(C=svm_C, kernel=svm_kernel, gamma=svm_gamma, random_state=random_state)
                elif svm_kernel == "poly":
                    svm_model = SVC(C=svm_C, kernel=svm_kernel, degree=svm_degree, random_state=random_state)
                else:
                    svm_model = SVC(C=svm_C, kernel=svm_kernel, random_state=random_state)
                
                # 训练SVM模型
                svm_model.fit(X_train, y_train)
                
                # 计算准确率
                svm_train_acc = accuracy_score(y_train, svm_model.predict(X_train))
                svm_test_acc = accuracy_score(y_test, svm_model.predict(X_test))
                
                # 绘制决策边界
                fig, ax = plt.subplots(figsize=(10, 6))
                kernel_params = ""
                if svm_kernel == "rbf":
                    kernel_params = f", gamma={svm_gamma}"
                elif svm_kernel == "poly":
                    kernel_params = f", degree={svm_degree}"
                
                plot_decision_boundary(svm_model, X_scaled, y, ax=ax, 
                                      title=f"SVM决策边界 (核函数={svm_kernel}, C={svm_C}{kernel_params})\n训练准确率: {svm_train_acc:.2f}, 测试准确率: {svm_test_acc:.2f}")
                st.pyplot(fig)
                
            else:  # 两者对比
                # 训练逻辑回归模型
                lr_model = LogisticRegression(C=lr_C, random_state=random_state)
                lr_model.fit(X_train, y_train)
                
                # 训练SVM模型
                if svm_kernel == "rbf":
                    svm_model = SVC(C=svm_C, kernel=svm_kernel, gamma=svm_gamma, random_state=random_state)
                elif svm_kernel == "poly":
                    svm_model = SVC(C=svm_C, kernel=svm_kernel, degree=svm_degree, random_state=random_state)
                else:
                    svm_model = SVC(C=svm_C, kernel=svm_kernel, random_state=random_state)
                
                svm_model.fit(X_train, y_train)
                
                # 计算准确率
                lr_train_acc = accuracy_score(y_train, lr_model.predict(X_train))
                lr_test_acc = accuracy_score(y_test, lr_model.predict(X_test))
                
                svm_train_acc = accuracy_score(y_train, svm_model.predict(X_train))
                svm_test_acc = accuracy_score(y_test, svm_model.predict(X_test))
                
                # 绘制对比图
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                plot_decision_boundary(lr_model, X_scaled, y, ax=ax1, 
                                      title=f"逻辑回归 (C={lr_C})\n训练: {lr_train_acc:.2f}, 测试: {lr_test_acc:.2f}")
                
                kernel_params = ""
                if svm_kernel == "rbf":
                    kernel_params = f", gamma={svm_gamma}"
                elif svm_kernel == "poly":
                    kernel_params = f", degree={svm_degree}"
                
                plot_decision_boundary(svm_model, X_scaled, y, ax=ax2, 
                                      title=f"SVM ({svm_kernel}, C={svm_C}{kernel_params})\n训练: {svm_train_acc:.2f}, 测试: {svm_test_acc:.2f}")
                
                st.pyplot(fig)
    
    # 添加决策边界的解释
    st.markdown("""
    ### 决策边界解释
    
    **决策边界**是分类算法用来区分不同类别的边界线。在上图中：
    - 红色区域表示模型预测为类别1的区域
    - 蓝色区域表示模型预测为类别0的区域
    - 散点表示实际数据样本，颜色对应其真实类别
    
    通过调整模型参数，您可以观察决策边界的变化，以及这些变化如何影响模型性能：
    
    - **正则化强度C**：较小的C值会使模型更简单（更正则化），边界更平滑；较大的C值会使模型更复杂，可能更好地拟合训练数据但可能过拟合
    - **核函数**：
      - 线性核(linear)产生线性决策边界
      - RBF核产生更复杂、非线性的边界，适合圆形或复杂结构
      - 多项式核(poly)可以产生非线性但相对平滑的边界
    
    尝试不同类型的数据集和参数组合，观察各算法的表现差异。
    """) 