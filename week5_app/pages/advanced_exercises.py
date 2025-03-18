import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os

# 添加utils目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.fonts import configure_matplotlib_fonts

def show():
    # 配置matplotlib字体
    configure_matplotlib_fonts()
    
    st.markdown("## 高级练习")
    
    st.markdown("### 正则化回归与超参数调优")
    
    st.markdown("""
    在本练习中，我们将深入探索正则化回归技术，包括Ridge回归（L2正则化）、Lasso回归（L1正则化）和ElasticNet回归（L1+L2混合正则化），
    并学习如何使用交叉验证和网格搜索进行超参数调优。
    
    **学习目标**：
    1. 掌握正则化回归的原理和应用
    2. 学习如何选择正则化系数
    3. 使用交叉验证和网格搜索进行超参数调优
    4. 对比不同正则化方法的性能
    """)
    
    # 加载数据
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    feature_names = diabetes.feature_names
    
    # 创建DataFrame方便查看
    diabetes_df = pd.DataFrame(X, columns=feature_names)
    diabetes_df['target'] = y
    
    # 显示数据集
    st.markdown("#### 数据预览：糖尿病数据集")
    st.markdown("""
    这个数据集包含了442个糖尿病患者的10个基线变量和一个一年后的疾病进展度量(目标变量)。
    变量经过了中心化和缩放处理。
    """)
    st.dataframe(diabetes_df.head())
    
    st.markdown("#### 数据描述")
    st.write(diabetes_df.describe())
    
    # 特征相关性
    st.markdown("#### 特征相关性分析")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = diabetes_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax)
    ax.set_title("特征相关性矩阵")
    st.pyplot(fig)
    
    # 模型对比实验
    st.markdown("### 正则化模型比较")
    
    st.markdown("""
    我们将对比以下回归模型：
    1. 线性回归（无正则化）
    2. Ridge回归（L2正则化）
    3. Lasso回归（L1正则化）
    4. ElasticNet回归（L1+L2混合正则化）
    
    对于每个模型，我们将使用交叉验证来选择最佳的正则化参数。
    """)
    
    # 数据准备
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 交互式参数调整
    st.markdown("#### 交互式参数调整")
    
    col1, col2 = st.columns(2)
    
    with col1:
        alpha_ridge = st.slider("Ridge正则化系数 (alpha)", 0.01, 10.0, 1.0, 0.01)
        alpha_lasso = st.slider("Lasso正则化系数 (alpha)", 0.001, 1.0, 0.1, 0.001)
    
    with col2:
        alpha_elastic = st.slider("ElasticNet正则化系数 (alpha)", 0.001, 1.0, 0.1, 0.001)
        l1_ratio = st.slider("ElasticNet L1比例 (l1_ratio)", 0.0, 1.0, 0.5, 0.01)
    
    # 训练模型
    models = {
        "线性回归": LinearRegression(),
        "Ridge回归": Ridge(alpha=alpha_ridge),
        "Lasso回归": Lasso(alpha=alpha_lasso, max_iter=10000),
        "ElasticNet回归": ElasticNet(alpha=alpha_elastic, l1_ratio=l1_ratio, max_iter=10000)
    }
    
    results = {}
    
    for name, model in models.items():
        # 训练
        model.fit(X_train_scaled, y_train)
        
        # 预测
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # 评估
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        results[name] = {
            "训练集 R²": train_r2,
            "测试集 R²": test_r2,
            "训练集 MSE": train_mse,
            "测试集 MSE": test_mse,
            "模型": model
        }
    
    # 显示结果
    st.markdown("#### 模型性能对比")
    
    result_df = pd.DataFrame({
        "模型": list(results.keys()),
        "训练集 R²": [results[k]["训练集 R²"] for k in results.keys()],
        "测试集 R²": [results[k]["测试集 R²"] for k in results.keys()],
        "训练集 MSE": [results[k]["训练集 MSE"] for k in results.keys()],
        "测试集 MSE": [results[k]["测试集 MSE"] for k in results.keys()]
    })
    
    st.dataframe(result_df)
    
    # 可视化结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    models_list = list(results.keys())
    train_r2_list = [results[k]["训练集 R²"] for k in models_list]
    test_r2_list = [results[k]["测试集 R²"] for k in models_list]
    
    ax1.bar(np.arange(len(models_list)), train_r2_list, width=0.4, label='训练集', alpha=0.7)
    ax1.bar(np.arange(len(models_list))+0.4, test_r2_list, width=0.4, label='测试集', alpha=0.7)
    ax1.set_xticks(np.arange(len(models_list))+0.2)
    ax1.set_xticklabels(models_list, rotation=45, ha='right')
    ax1.set_ylabel('R²')
    ax1.set_title('不同模型的R²对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    train_mse_list = [results[k]["训练集 MSE"] for k in models_list]
    test_mse_list = [results[k]["测试集 MSE"] for k in models_list]
    
    ax2.bar(np.arange(len(models_list)), train_mse_list, width=0.4, label='训练集', alpha=0.7)
    ax2.bar(np.arange(len(models_list))+0.4, test_mse_list, width=0.4, label='测试集', alpha=0.7)
    ax2.set_xticks(np.arange(len(models_list))+0.2)
    ax2.set_xticklabels(models_list, rotation=45, ha='right')
    ax2.set_ylabel('MSE')
    ax2.set_title('不同模型的MSE对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 系数分析
    st.markdown("#### 特征系数分析")
    
    # 获取每个模型的系数
    coef_df = pd.DataFrame({"特征": feature_names})
    
    for name, result in results.items():
        if hasattr(result["模型"], "coef_"):
            coef_df[name] = result["模型"].coef_
    
    st.dataframe(coef_df)
    
    # 绘制系数对比图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 创建热图
    coef_array = coef_df.iloc[:, 1:].values
    sns.heatmap(coef_array, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=coef_df.columns[1:], yticklabels=feature_names, ax=ax)
    ax.set_title("不同模型的特征系数对比")
    ax.set_xlabel("模型")
    ax.set_ylabel("特征")
    
    st.pyplot(fig)
    
    st.markdown("""
    **观察**:
    
    1. **正则化效果**:
       - Ridge保留了大部分特征，但减小了系数值
       - Lasso倾向于产生稀疏解，将部分系数降为0
       - ElasticNet结合了两者的特性
       
    2. **特征重要性**:
       - 不同的正则化方法可能会改变特征的相对重要性
       - Lasso提供了一种内置的特征选择机制
    """)
    
    # 超参数调优
    st.markdown("### 超参数调优")
    
    st.markdown("""
    我们将使用GridSearchCV进行超参数的网格搜索，找到最佳的正则化参数。
    
    **注意**: 网格搜索可能需要一些时间来运行，尤其是参数空间很大时。
    """)
    
    if st.button("运行超参数网格搜索"):
        st.info("正在执行网格搜索，这可能需要一些时间...")
        
        # 定义参数网格
        param_grid = {
            "Ridge": {"alpha": np.logspace(-3, 3, 20)},
            "Lasso": {"alpha": np.logspace(-4, 0, 20)},
            "ElasticNet": {
                "alpha": np.logspace(-3, 0, 10),
                "l1_ratio": np.linspace(0.1, 0.9, 9)
            }
        }
        
        best_params = {}
        cv_results = {}
        
        # Ridge超参数搜索
        ridge_cv = GridSearchCV(
            Ridge(), param_grid["Ridge"], cv=5, scoring="neg_mean_squared_error"
        )
        ridge_cv.fit(X_train_scaled, y_train)
        best_params["Ridge"] = ridge_cv.best_params_
        cv_results["Ridge"] = ridge_cv.cv_results_
        
        # Lasso超参数搜索
        lasso_cv = GridSearchCV(
            Lasso(max_iter=10000), param_grid["Lasso"], cv=5, scoring="neg_mean_squared_error"
        )
        lasso_cv.fit(X_train_scaled, y_train)
        best_params["Lasso"] = lasso_cv.best_params_
        cv_results["Lasso"] = lasso_cv.cv_results_
        
        # ElasticNet超参数搜索
        elastic_cv = GridSearchCV(
            ElasticNet(max_iter=10000), param_grid["ElasticNet"], cv=5, scoring="neg_mean_squared_error"
        )
        elastic_cv.fit(X_train_scaled, y_train)
        best_params["ElasticNet"] = elastic_cv.best_params_
        cv_results["ElasticNet"] = elastic_cv.cv_results_
        
        # 显示最佳参数
        st.markdown("#### 网格搜索最佳参数")
        for model_name, params in best_params.items():
            st.write(f"**{model_name}最佳参数:** {params}")
        
        # 使用最佳参数训练模型
        best_models = {
            "Ridge": Ridge(**best_params["Ridge"]),
            "Lasso": Lasso(**best_params["Lasso"], max_iter=10000),
            "ElasticNet": ElasticNet(**best_params["ElasticNet"], max_iter=10000)
        }
        
        best_results = {}
        
        for name, model in best_models.items():
            # 训练
            model.fit(X_train_scaled, y_train)
            
            # 预测
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # 评估
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            
            best_results[name] = {
                "训练集 R²": train_r2,
                "测试集 R²": test_r2,
                "训练集 MSE": train_mse,
                "测试集 MSE": test_mse
            }
        
        # 显示最佳模型结果
        st.markdown("#### 最佳模型性能对比")
        
        best_result_df = pd.DataFrame({
            "模型": list(best_results.keys()),
            "训练集 R²": [best_results[k]["训练集 R²"] for k in best_results.keys()],
            "测试集 R²": [best_results[k]["测试集 R²"] for k in best_results.keys()],
            "训练集 MSE": [best_results[k]["训练集 MSE"] for k in best_results.keys()],
            "测试集 MSE": [best_results[k]["测试集 MSE"] for k in best_results.keys()]
        })
        
        st.dataframe(best_result_df)
        
        # 可视化参数对性能的影响
        st.markdown("#### 正则化系数对模型性能的影响")
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Ridge
        alpha_values = param_grid["Ridge"]["alpha"]
        mean_test_scores = -cv_results["Ridge"]["mean_test_score"]
        std_test_scores = cv_results["Ridge"]["std_test_score"]
        
        ax1.semilogx(alpha_values, mean_test_scores)
        ax1.fill_between(alpha_values, mean_test_scores - std_test_scores,
                        mean_test_scores + std_test_scores, alpha=0.3)
        ax1.set_xlabel("alpha")
        ax1.set_ylabel("Mean Squared Error")
        ax1.set_title("Ridge: alpha vs MSE")
        ax1.axvline(x=best_params["Ridge"]["alpha"], color='r', linestyle='--')
        ax1.grid(True, alpha=0.3)
        
        # Lasso
        alpha_values = param_grid["Lasso"]["alpha"]
        mean_test_scores = -cv_results["Lasso"]["mean_test_score"]
        std_test_scores = cv_results["Lasso"]["std_test_score"]
        
        ax2.semilogx(alpha_values, mean_test_scores)
        ax2.fill_between(alpha_values, mean_test_scores - std_test_scores,
                        mean_test_scores + std_test_scores, alpha=0.3)
        ax2.set_xlabel("alpha")
        ax2.set_ylabel("Mean Squared Error")
        ax2.set_title("Lasso: alpha vs MSE")
        ax2.axvline(x=best_params["Lasso"]["alpha"], color='r', linestyle='--')
        ax2.grid(True, alpha=0.3)
        
        # ElasticNet - 只显示最佳l1_ratio下的alpha变化
        best_l1_ratio = best_params["ElasticNet"]["l1_ratio"]
        alpha_values = param_grid["ElasticNet"]["alpha"]
        
        # 筛选最佳l1_ratio对应的结果
        mask = [
            params["l1_ratio"] == best_l1_ratio 
            for params in cv_results["ElasticNet"]["params"]
        ]
        filtered_results = {
            "params": np.array(cv_results["ElasticNet"]["params"])[mask],
            "mean_test_score": np.array(cv_results["ElasticNet"]["mean_test_score"])[mask],
            "std_test_score": np.array(cv_results["ElasticNet"]["std_test_score"])[mask]
        }
        
        # 提取alpha值和对应的分数
        filtered_alphas = [params["alpha"] for params in filtered_results["params"]]
        filtered_scores = -filtered_results["mean_test_score"]
        filtered_stds = filtered_results["std_test_score"]
        
        ax3.semilogx(filtered_alphas, filtered_scores)
        ax3.fill_between(filtered_alphas, filtered_scores - filtered_stds,
                        filtered_scores + filtered_stds, alpha=0.3)
        ax3.set_xlabel("alpha (at l1_ratio = {:.2f})".format(best_l1_ratio))
        ax3.set_ylabel("Mean Squared Error")
        ax3.set_title("ElasticNet: alpha vs MSE")
        ax3.axvline(x=best_params["ElasticNet"]["alpha"], color='r', linestyle='--')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **解读**:
        
        1. **正则化强度 (alpha)**:
           - 过小的alpha可能导致过拟合
           - 过大的alpha可能导致欠拟合
           - 红色虚线标记了网格搜索找到的最佳alpha值
           
        2. **模型选择**:
           - 根据网格搜索结果，可以选择性能最好的模型进行最终预测
           - 不同的评估指标可能会导致不同的模型选择
        """)
    
    # 实际项目挑战
    st.markdown("### 实际项目挑战")
    
    st.markdown("""
    **任务**: 构建一个完整的回归分析流程，应用到真实数据集。
    
    **步骤**:
    
    1. **数据选择与预处理**:
       - 选择一个感兴趣的回归数据集（如波士顿房价、加州房价、自行车共享等）
       - 进行数据清洗、特征工程和预处理
       
    2. **模型构建与评估**:
       - 实现并对比多种回归模型（线性回归、多项式回归、Ridge、Lasso、ElasticNet）
       - 使用交叉验证进行超参数调优
       - 评估模型性能并解释结果
       
    3. **特征重要性分析**:
       - 分析哪些特征对预测最重要
       - 使用正则化技术进行特征筛选
       
    4. **模型解释与可视化**:
       - 创建直观的可视化展示模型结果
       - 解释模型系数的实际含义
    
    **提示**: 可以使用Scikit-learn提供的数据集，也可以从Kaggle等平台获取真实数据集。
    """)
    
    # 进一步学习资源
    st.markdown("### 进一步学习资源")
    
    st.markdown("""
    1. **scikit-learn文档**:
       - [线性模型](https://scikit-learn.org/stable/modules/linear_model.html)
       - [交叉验证](https://scikit-learn.org/stable/modules/cross_validation.html)
       - [超参数调优](https://scikit-learn.org/stable/modules/grid_search.html)
    
    2. **推荐书籍**:
       - 《Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow》
       - 《Python Machine Learning》- Sebastian Raschka
    
    3. **在线课程**:
       - Coursera: [吴恩达机器学习课程](https://www.coursera.org/learn/machine-learning)
       - edX: [Microsoft的数据科学课程](https://www.edx.org/professional-certificate/microsoft-data-science)
    
    4. **进阶主题**:
       - 特征选择技术
       - 高级正则化方法
       - 非线性回归技术（如支持向量回归、随机森林回归）
    """)

if __name__ == "__main__":
    show() 