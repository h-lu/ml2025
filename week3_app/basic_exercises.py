import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import seaborn as sns
import time
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

def load_data():
    """加载乳腺癌数据集"""
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names
    
    return X, y, feature_names, target_names

def plot_confusion_matrix(cm, class_names):
    """绘制混淆矩阵热图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.title('混淆矩阵')
    return fig

def plot_roc_curve(y_true, y_pred_proba, model_name):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('假正例率 (FPR)')
    ax.set_ylabel('真正例率 (TPR)')
    ax.set_title('接收者操作特征曲线 (ROC)')
    ax.legend(loc='lower right')
    
    return fig

def exercise_1():
    """练习1: 乳腺癌数据分类"""
    st.subheader("练习1: 乳腺癌数据分类")
    
    st.markdown("""
    在这个练习中，我们将使用sklearn自带的乳腺癌数据集来训练和评估分类模型。
    这个数据集包含569个样本，每个样本有30个特征，目标是将肿瘤分类为良性(benign)或恶性(malignant)。
    """)
    
    # 加载数据
    X, y, feature_names, target_names = load_data()
    
    # 显示数据集信息
    st.markdown(f"**数据集信息**：\n- 样本数: {X.shape[0]}\n- 特征数: {X.shape[1]}")
    st.markdown(f"**类别分布**：\n- {target_names[0]} (0): {np.sum(y == 0)}\n- {target_names[1]} (1): {np.sum(y == 1)}")
    
    # 数据预处理选项
    st.markdown("### 数据预处理")
    
    # 测试集比例选择
    test_size = st.slider("测试集比例", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    
    # 随机种子
    random_state = st.slider("随机种子", min_value=0, max_value=100, value=42, step=1)
    
    # 是否标准化数据
    do_scaling = st.checkbox("标准化特征", value=True)
    
    # 模型选择
    st.markdown("### 模型选择")
    model_type = st.radio("选择模型", ["逻辑回归", "SVM", "两者都训练"])
    
    # 逻辑回归参数
    if model_type in ["逻辑回归", "两者都训练"]:
        st.markdown("#### 逻辑回归参数")
        lr_C = st.slider("正则化强度C", min_value=0.01, max_value=10.0, value=1.0, step=0.1, key="lr_C")
        lr_max_iter = st.slider("最大迭代次数", min_value=100, max_value=2000, value=1000, step=100, key="lr_iter")
        lr_solver = st.selectbox("优化算法", ["liblinear", "lbfgs", "newton-cg", "saga"])
    
    # SVM参数
    if model_type in ["SVM", "两者都训练"]:
        st.markdown("#### SVM参数")
        svm_C = st.slider("正则化强度C", min_value=0.01, max_value=10.0, value=1.0, step=0.1, key="svm_C")
        svm_kernel = st.selectbox("核函数", ["linear", "rbf", "poly"])
        
        if svm_kernel == "rbf":
            svm_gamma = st.slider("Gamma参数", min_value=0.001, max_value=1.0, value=0.1, step=0.01)
    
    # 训练模型按钮
    train_btn = st.button("训练模型")
    
    # 训练和评估模型
    if train_btn:
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        
        # 标准化数据
        if do_scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        # 训练模型并评估
        if model_type in ["逻辑回归", "两者都训练"]:
            # 开始计时
            start_time = time.time()
            
            # 训练逻辑回归模型
            lr_model = LogisticRegression(C=lr_C, max_iter=lr_max_iter, solver=lr_solver, random_state=random_state)
            lr_model.fit(X_train, y_train)
            
            # 模型预测
            lr_pred = lr_model.predict(X_test)
            lr_pred_proba = lr_model.predict_proba(X_test)[:,1]
            
            # 计算评估指标
            lr_acc = accuracy_score(y_test, lr_pred)
            lr_cm = confusion_matrix(y_test, lr_pred)
            lr_report = classification_report(y_test, lr_pred, target_names=target_names)
            
            # 训练时间
            lr_time = time.time() - start_time
            
            # 显示逻辑回归结果
            st.markdown("### 逻辑回归模型结果")
            st.markdown(f"**训练时间**: {lr_time:.4f} 秒")
            st.markdown(f"**测试集准确率**: {lr_acc:.4f}")
            
            # 显示混淆矩阵
            st.markdown("**混淆矩阵**:")
            st.pyplot(plot_confusion_matrix(lr_cm, target_names))
            
            # 显示分类报告
            st.markdown("**分类报告**:")
            st.text(lr_report)
            
            # 显示ROC曲线
            st.markdown("**ROC曲线**:")
            st.pyplot(plot_roc_curve(y_test, lr_pred_proba, "逻辑回归"))
            
            # 输出模型系数
            if st.checkbox("查看逻辑回归模型系数", value=False):
                coefficients = pd.DataFrame({
                    '特征': feature_names,
                    '系数': lr_model.coef_[0]
                }).sort_values('系数', ascending=False)
                
                st.dataframe(coefficients)
                
                # 绘制系数条形图
                fig, ax = plt.subplots(figsize=(10, 8))
                coefficients.plot(x='特征', y='系数', kind='bar', ax=ax)
                plt.xticks(rotation=90)
                plt.title('逻辑回归模型系数')
                plt.tight_layout()
                st.pyplot(fig)
        
        if model_type in ["SVM", "两者都训练"]:
            # 开始计时
            start_time = time.time()
            
            # 训练SVM模型
            if svm_kernel == "rbf":
                svm_model = SVC(C=svm_C, kernel=svm_kernel, gamma=svm_gamma, probability=True, random_state=random_state)
            else:
                svm_model = SVC(C=svm_C, kernel=svm_kernel, probability=True, random_state=random_state)
            
            svm_model.fit(X_train, y_train)
            
            # 模型预测
            svm_pred = svm_model.predict(X_test)
            svm_pred_proba = svm_model.predict_proba(X_test)[:,1]
            
            # 计算评估指标
            svm_acc = accuracy_score(y_test, svm_pred)
            svm_cm = confusion_matrix(y_test, svm_pred)
            svm_report = classification_report(y_test, svm_pred, target_names=target_names)
            
            # 训练时间
            svm_time = time.time() - start_time
            
            # 显示SVM结果
            st.markdown("### SVM模型结果")
            st.markdown(f"**训练时间**: {svm_time:.4f} 秒")
            st.markdown(f"**测试集准确率**: {svm_acc:.4f}")
            
            # 显示混淆矩阵
            st.markdown("**混淆矩阵**:")
            st.pyplot(plot_confusion_matrix(svm_cm, target_names))
            
            # 显示分类报告
            st.markdown("**分类报告**:")
            st.text(svm_report)
            
            # 显示ROC曲线
            st.markdown("**ROC曲线**:")
            st.pyplot(plot_roc_curve(y_test, svm_pred_proba, "SVM"))
        
        # 如果两种模型都训练了，显示比较
        if model_type == "两者都训练":
            st.markdown("### 模型比较")
            
            # 比较准确率
            st.markdown("**准确率比较**")
            acc_data = pd.DataFrame({
                '模型': ['逻辑回归', 'SVM'],
                '准确率': [lr_acc, svm_acc],
                '训练时间(秒)': [lr_time, svm_time]
            })
            st.table(acc_data)
            
            # 绘制ROC曲线比较
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # 逻辑回归ROC
            fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_pred_proba)
            auc_lr = roc_auc_score(y_test, lr_pred_proba)
            ax.plot(fpr_lr, tpr_lr, label=f'逻辑回归 (AUC = {auc_lr:.3f})')
            
            # SVM ROC
            fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_pred_proba)
            auc_svm = roc_auc_score(y_test, svm_pred_proba)
            ax.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {auc_svm:.3f})')
            
            # 对角线
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('假正例率 (FPR)')
            ax.set_ylabel('真正例率 (TPR)')
            ax.set_title('ROC曲线比较')
            ax.legend(loc='lower right')
            
            st.pyplot(fig)
    
    # 练习问题
    st.markdown("### 练习问题")
    
    st.markdown("""
    1. 观察逻辑回归模型的系数，哪些特征对预测结果影响最大？这些特征在医学上有什么含义？
    
    2. 尝试调整模型参数（如正则化强度C），观察模型性能的变化。太小或太大的C值会导致什么问题？
    
    3. 对于这个数据集，逻辑回归和SVM哪个表现更好？为什么？
    
    4. 混淆矩阵中的假阳性和假阴性哪个更严重？在医疗诊断场景中，如何权衡这两种错误？
    
    5. 如果要进一步提高模型性能，你会尝试哪些方法？
    """)

def exercise_2():
    """练习2: 鸢尾花分类（多分类问题）"""
    st.subheader("练习2: 鸢尾花分类（多分类问题）")
    
    st.markdown("""
    在这个练习中，我们将使用经典的鸢尾花(Iris)数据集，这是一个多分类问题。
    我们将扩展我们的知识，看看逻辑回归和SVM如何处理多类别分类任务。
    """)
    
    st.info("多分类问题是分类任务的自然扩展，其中目标变量可以取两个以上的离散值。")
    
    # 实现说明
    st.markdown("""
    **实现说明**:
    
    1. 鸢尾花数据集包含3个类别，每个类别50个样本，每个样本有4个特征（花萼长度、花萼宽度、花瓣长度、花瓣宽度）
    
    2. 对于逻辑回归，scikit-learn使用"一对多"(OvR)策略处理多分类问题
    
    3. 对于SVM，根据所选核函数和设置，可以使用"一对一"(OvO)或"一对多"(OvR)策略
    
    4. 你将有机会调整参数，观察它们对多分类性能的影响
    """)
    
    # 练习内容提示
    st.markdown("""
    **提示**: 尝试绘制鸢尾花数据的散点图，观察特征之间的关系，这将帮助你理解数据结构。
    
    **挑战**: 尝试结合主成分分析(PCA)来降维并可视化决策边界。
    """)
    
    # 实现代码位于完整练习中
    
    # 添加交互式鸢尾花分类练习
    
    # 加载鸢尾花数据集
    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # 数据探索
    st.markdown("### 数据探索")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**数据集概况**")
        st.write(f"样本数量: {X_iris.shape[0]}")
        st.write(f"特征数量: {X_iris.shape[1]}")
        st.write(f"类别数量: {len(target_names)}")
        
        # 显示类别分布
        class_dist = pd.Series(y_iris).value_counts().sort_index()
        st.bar_chart(class_dist)
    
    with col2:
        st.markdown("**特征描述**")
        feature_descriptions = pd.DataFrame({
            "特征名称": feature_names,
            "单位": ["厘米"] * 4,
            "描述": [
                "花萼长度 (Sepal Length)",
                "花萼宽度 (Sepal Width)",
                "花瓣长度 (Petal Length)",
                "花瓣宽度 (Petal Width)"
            ]
        })
        st.dataframe(feature_descriptions)
    
    # 特征选择和数据可视化
    st.markdown("### 特征可视化")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_feature = st.selectbox("选择X轴特征", feature_names, index=0)
        y_feature = st.selectbox("选择Y轴特征", feature_names, index=2)
    
    with col2:
        use_pca = st.checkbox("使用PCA降维", value=False)
        color_by = st.radio("颜色标记", ["类别", "统一颜色"])
    
    # 绘制散点图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if not use_pca:
        # 获取所选特征的索引
        x_idx = feature_names.index(x_feature)
        y_idx = feature_names.index(y_feature)
        
        # 绘制散点图
        if color_by == "类别":
            for i, target in enumerate(target_names):
                ax.scatter(
                    X_iris[y_iris == i, x_idx],
                    X_iris[y_iris == i, y_idx],
                    label=target
                )
        else:
            ax.scatter(X_iris[:, x_idx], X_iris[:, y_idx])
        
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_title(f"{x_feature} vs {y_feature}")
        
        if color_by == "类别":
            ax.legend()
    else:
        # 使用PCA降维
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_iris)
        
        # 绘制PCA结果
        if color_by == "类别":
            for i, target in enumerate(target_names):
                ax.scatter(
                    X_pca[y_iris == i, 0],
                    X_pca[y_iris == i, 1],
                    label=target
                )
        else:
            ax.scatter(X_pca[:, 0], X_pca[:, 1])
        
        ax.set_xlabel("主成分1")
        ax.set_ylabel("主成分2")
        ax.set_title("PCA降维结果")
        
        # 显示方差解释比例
        explained_variance = pca.explained_variance_ratio_
        st.write(f"主成分1解释方差比例: {explained_variance[0]:.2f}")
        st.write(f"主成分2解释方差比例: {explained_variance[1]:.2f}")
        st.write(f"总解释方差比例: {sum(explained_variance):.2f}")
        
        if color_by == "类别":
            ax.legend()
    
    # 显示图表
    st.pyplot(fig)
    
    # 模型训练部分
    st.markdown("### 模型训练")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.radio(
            "选择模型",
            ["逻辑回归", "SVM"]
        )
        
        test_size = st.slider("测试集比例", 0.1, 0.5, 0.3, 0.05)
        random_state = st.slider("随机种子", 0, 100, 42, 1)
    
    with col2:
        if model_type == "逻辑回归":
            C = st.slider("正则化强度 (C)", 0.01, 10.0, 1.0, 0.1)
            multi_class = st.radio("多分类策略", ["ovr", "multinomial"])
            solver = st.selectbox("优化器", ["lbfgs", "newton-cg", "sag", "saga"])
        else:  # SVM
            C = st.slider("正则化强度 (C)", 0.01, 10.0, 1.0, 0.1)
            kernel = st.selectbox("核函数", ["linear", "rbf", "poly"])
            
            if kernel == "rbf":
                gamma = st.slider("Gamma参数", 0.01, 10.0, 0.1, 0.01)
            elif kernel == "poly":
                degree = st.slider("多项式阶数", 1, 5, 3, 1)
    
    # 训练和评估模型
    if st.button("训练模型", key="iris_train"):
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=test_size, random_state=random_state, stratify=y_iris)
        
        # 标准化数据
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 训练模型
        if model_type == "逻辑回归":
            model = LogisticRegression(C=C, multi_class=multi_class, solver=solver, max_iter=1000, random_state=random_state)
        else:  # SVM
            if kernel == "rbf":
                model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=random_state)
            elif kernel == "poly":
                model = SVC(C=C, kernel=kernel, degree=degree, probability=True, random_state=random_state)
            else:
                model = SVC(C=C, kernel=kernel, probability=True, random_state=random_state)
        
        # 计时开始
        start_time = time.time()
        
        # 拟合模型
        model.fit(X_train_scaled, y_train)
        
        # 预测
        y_pred = model.predict(X_test_scaled)
        
        # 计算性能指标
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=target_names)
        
        # 计时结束
        training_time = time.time() - start_time
        
        # 显示结果
        st.markdown("### 模型评估结果")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("准确率", f"{accuracy:.4f}")
        with col2:
            st.metric("训练时间", f"{training_time:.4f}秒")
        with col3:
            n_support = getattr(model, "n_support_", None)
            if n_support is not None:
                st.metric("支持向量数量", sum(n_support))
        
        # 混淆矩阵
        st.markdown("**混淆矩阵**")
        fig_cm = plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.title('混淆矩阵')
        st.pyplot(fig_cm)
        
        # 分类报告
        st.markdown("**分类报告**")
        st.text(report)
        
        # 如果使用PCA，绘制决策边界
        if use_pca:
            st.markdown("### 决策边界可视化 (PCA空间)")
            
            # 创建网格点
            h = 0.02  # 网格步长
            x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
            y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            
            # 创建一个新的模型，在PCA空间中训练
            if model_type == "逻辑回归":
                pca_model = LogisticRegression(C=C, multi_class=multi_class, solver=solver, max_iter=1000, random_state=random_state)
            else:  # SVM
                if kernel == "rbf":
                    pca_model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=random_state)
                elif kernel == "poly":
                    pca_model = SVC(C=C, kernel=kernel, degree=degree, random_state=random_state)
                else:
                    pca_model = SVC(C=C, kernel=kernel, random_state=random_state)
            
            pca_model.fit(X_pca, y_iris)
            
            # 预测网格点
            Z = pca_model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # 绘制决策边界
            fig_boundary, ax = plt.subplots(figsize=(10, 8))
            ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
            
            # 绘制原始数据点
            for i, target in enumerate(target_names):
                ax.scatter(
                    X_pca[y_iris == i, 0],
                    X_pca[y_iris == i, 1],
                    label=target
                )
            
            ax.set_xlabel("主成分1")
            ax.set_ylabel("主成分2")
            ax.set_title(f"{model_type}在PCA空间的决策边界")
            ax.legend()
            
            st.pyplot(fig_boundary)
    
    # 练习讨论与结论
    st.markdown("### 思考问题")
    
    st.markdown("""
    **讨论以下问题**:
    
    1. 鸢尾花数据集中，哪些特征对分类最有帮助？这可以从散点图和PCA分析中得出吗？
    
    2. 逻辑回归和SVM在这个多分类问题上的表现有什么不同？为什么？
    
    3. 多分类策略（OvR和multinomial）如何影响逻辑回归的性能和决策边界？
    
    4. 不同核函数如何改变SVM的决策边界和分类性能？
    
    5. 在这个数据集上，如何选择最佳的超参数？除了我们已经尝试的参数外，还有哪些可能会影响性能？
    """)

def show_basic_exercises():
    """显示基础练习页面"""
    
    st.header("基础练习")
    
    st.markdown("""
    本节包含一些基础练习，帮助你理解和实践逻辑回归和SVM分类算法。
    这些练习设计为循序渐进的学习体验，从简单的二分类问题开始，然后扩展到多分类场景。
    """)
    
    # 选择练习
    exercise = st.radio(
        "选择一个练习:",
        ["练习1: 乳腺癌数据分类", "练习2: 鸢尾花分类（多分类问题）"]
    )
    
    if exercise == "练习1: 乳腺癌数据分类":
        exercise_1()
    else:
        exercise_2() 