import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import sys
import os
from io import BytesIO
import base64

# 导入工具函数
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_utils import generate_synthetic_data, split_and_preprocess_data
from utils.model_utils import train_xgboost_model, evaluate_model, plot_feature_importance
# 导入matplotlib中文字体支持和图像处理函数
from utils.plot_utils import configure_matplotlib_fonts, create_chinese_text_image, fig_to_base64, get_chinese_plot

# 配置matplotlib支持中文
configure_matplotlib_fonts()

def show():
    # 确保在函数内部再次调用字体配置，保证每次页面加载都有正确配置
    configure_matplotlib_fonts()
    
    st.title("代码演示: XGBoost实现与调优")
    
    # 创建示例数据
    with st.expander("数据生成与准备", expanded=True):
        st.markdown("""
        ### 数据生成与准备
        
        下面我们将生成一个包含非线性关系的模拟数据集，用于演示XGBoost模型的性能。
        """)
        
        # 用户可以调整样本数量
        n_samples = st.slider("样本数量", 100, 2000, 1000)
        
        # 用户可以选择添加噪声程度
        noise_level = st.slider("噪声程度", 0.1, 2.0, 1.0)
        
        # 生成代码显示
        st.code("""
# 生成模拟数据
np.random.seed(42)
n_samples = 1000  # 样本数量
X = np.random.rand(n_samples, 5)  # 5个特征
# 创建非线性关系
y = 5 + 3*X[:, 0]**2 + 2*X[:, 1] + np.sin(3*X[:, 2]) + np.exp(X[:, 3]) - 2*X[:, 4] + np.random.normal(0, 1, n_samples)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
        """, language="python")
        
        # 实际生成数据
        np.random.seed(42)
        X = np.random.rand(n_samples, 5)
        y = 5 + 3*X[:, 0]**2 + 2*X[:, 1] + np.sin(3*X[:, 2]) + np.exp(X[:, 3]) - 2*X[:, 4] + np.random.normal(0, noise_level, n_samples)
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 显示数据分布
        st.markdown("### 生成的数据集分布")
        
        # 定义绘图函数
        def plot_data_distribution(ax):
            # 目标变量分布
            sns.histplot(y, kde=True, ax=ax)
            ax.set_title('Target Distribution')
        
        # 使用新的方法显示目标变量分布
        st.write(get_chinese_plot(plot_data_distribution, "目标变量分布", "取值", "频率"), unsafe_allow_html=True)
        
        # 显示各特征与目标变量的关系
        col1, col2 = st.columns(2)
        
        # 前三个特征
        with col1:
            for i in range(3):
                def plot_feature(ax, feature_idx=i):
                    ax.scatter(X[:, feature_idx], y, alpha=0.3)
                    ax.set_title(f'Feature {feature_idx+1} vs Target')
                    ax.set_xlabel(f'Feature {feature_idx+1}')
                    ax.set_ylabel('Target')
                
                st.write(get_chinese_plot(
                    lambda ax: plot_feature(ax, i), 
                    f"特征 {i+1} 与目标变量关系",
                    f"特征 {i+1}",
                    "目标变量"
                ), unsafe_allow_html=True)
        
        # 后两个特征
        with col2:
            for i in range(3, 5):
                def plot_feature(ax, feature_idx=i):
                    ax.scatter(X[:, feature_idx], y, alpha=0.3)
                    ax.set_title(f'Feature {feature_idx+1} vs Target')
                    ax.set_xlabel(f'Feature {feature_idx+1}')
                    ax.set_ylabel('Target')
                
                st.write(get_chinese_plot(
                    lambda ax: plot_feature(ax, i), 
                    f"特征 {i+1} 与目标变量关系",
                    f"特征 {i+1}",
                    "目标变量"
                ), unsafe_allow_html=True)
        
        # 使用HTML和纯文本显示数据集信息，避免中文显示问题
        st.markdown("""
        <h3 style='text-align:center;'>数据集信息</h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("训练集样本数", len(X_train))
        with col2:
            st.metric("测试集样本数", len(X_test))
        with col3:
            st.metric("特征数量", X.shape[1])

    # XGBoost模型训练
    with st.expander("基本XGBoost模型", expanded=True):
        st.markdown("""
        ### 基本XGBoost模型
        
        下面我们将训练一个基本的XGBoost回归模型，并与线性回归模型进行对比。
        """)
        
        # 线性回归作为基线
        st.markdown("#### 1. 线性回归基线模型")
        
        with st.echo():
            # 训练线性回归模型
            lr = LinearRegression()
            lr.fit(X_train_scaled, y_train)
            lr_pred = lr.predict(X_test_scaled)
            
            # 评估线性回归模型
            lr_mse = mean_squared_error(y_test, lr_pred)
            lr_rmse = np.sqrt(lr_mse)
            lr_mae = mean_absolute_error(y_test, lr_pred)
            lr_r2 = r2_score(y_test, lr_pred)
        
        # 显示线性回归结果
        st.markdown("**线性回归性能:**")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MSE", f"{lr_mse:.4f}")
        with col2:
            st.metric("RMSE", f"{lr_rmse:.4f}")
        with col3:
            st.metric("MAE", f"{lr_mae:.4f}")
        with col4:
            st.metric("R²", f"{lr_r2:.4f}")
        
        # XGBoost模型
        st.markdown("#### 2. 基本XGBoost模型")
        
        # 用户可调参数
        col1, col2 = st.columns(2)
        with col1:
            max_depth = st.slider("max_depth", 1, 10, 5)
            learning_rate = st.slider("learning_rate", 0.01, 0.5, 0.1)
        with col2:
            n_estimators = st.slider("n_estimators", 10, 200, 100)
            min_child_weight = st.slider("min_child_weight", 1, 10, 1)
        
        with st.echo():
            # 设置XGBoost参数
            params = {
                'objective': 'reg:squarederror',
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'n_estimators': n_estimators,
                'min_child_weight': min_child_weight,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'random_state': 42
            }
            
            # 训练XGBoost模型
            xgb_model = xgb.XGBRegressor(**params)
            xgb_model.fit(X_train_scaled, y_train)
            xgb_pred = xgb_model.predict(X_test_scaled)
            
            # 评估XGBoost模型
            xgb_mse = mean_squared_error(y_test, xgb_pred)
            xgb_rmse = np.sqrt(xgb_mse)
            xgb_mae = mean_absolute_error(y_test, xgb_pred)
            xgb_r2 = r2_score(y_test, xgb_pred)
        
        # 显示XGBoost结果
        st.markdown("**XGBoost模型性能:**")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MSE", f"{xgb_mse:.4f}", f"{(lr_mse - xgb_mse):.4f}")
        with col2:
            st.metric("RMSE", f"{xgb_rmse:.4f}", f"{(lr_rmse - xgb_rmse):.4f}")
        with col3:
            st.metric("MAE", f"{xgb_mae:.4f}", f"{(lr_mae - xgb_mae):.4f}")
        with col4:
            st.metric("R²", f"{xgb_r2:.4f}", f"{(xgb_r2 - lr_r2):.4f}")
        
        # 预测vs实际值
        st.markdown("#### 预测值 vs 实际值")
        
        # 使用新的绘图方式
        def plot_predictions(ax):
            # 分成两个子图
            ax1 = ax
            ax2 = ax.twinx()
            
            # 线性回归
            ax1.scatter(y_test, lr_pred, alpha=0.5, color='blue', label='线性回归')
            ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            
            # XGBoost
            ax2.scatter(y_test, xgb_pred, alpha=0.5, color='green', label='XGBoost')
            
            # 添加图例
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
        
        st.write(get_chinese_plot(
            plot_predictions, 
            "预测值 vs 实际值比较", 
            "实际值", 
            "预测值", 
            figsize=(10, 6)
        ), unsafe_allow_html=True)
    
    # 特征重要性分析
    with st.expander("特征重要性分析", expanded=True):
        st.markdown("### 特征重要性分析")
        
        st.markdown("""
        XGBoost提供了多种特征重要性度量方式:
        * weight: 特征在所有树中被用作分裂点的次数
        * gain: 特征带来的平均增益
        * cover: 特征覆盖的平均样本数量
        """)
        
        # 选择特征重要性类型
        importance_type = st.radio(
            "选择特征重要性类型",
            ["weight", "gain", "cover"],
            horizontal=True
        )
        
        # 计算并展示特征重要性
        feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
        
        # 获取特征重要性
        importance = xgb_model.get_booster().get_score(importance_type=importance_type)
        
        # 转换为DataFrame
        importance_df = pd.DataFrame({
            'Feature': list(importance.keys()),
            'Importance': list(importance.values())
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # 使用新的绘图方式
        def plot_feature_importance(ax):
            sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
        
        st.write(get_chinese_plot(
            plot_feature_importance, 
            f"XGBoost特征重要性 ({importance_type})", 
            "重要性", 
            "特征",
            figsize=(10, 6)
        ), unsafe_allow_html=True)
        
        # 显示表格
        st.markdown("#### 特征重要性表格")
        st.table(importance_df.set_index('Feature'))
        
    # 简单交叉验证
    with st.expander("模型交叉验证", expanded=False):
        st.markdown("### 模型交叉验证")
        
        from sklearn.model_selection import cross_val_score, KFold
        
        st.markdown("""
        交叉验证是评估模型性能稳定性的重要方法。这里我们使用K折交叉验证来评估XGBoost模型。
        """)
        
        # 设置折数
        n_folds = st.slider("交叉验证折数", 3, 10, 5)
        
        with st.echo():
            # 设置交叉验证
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            # XGBoost交叉验证
            xgb_cv_scores = cross_val_score(
                xgb_model, X_train_scaled, y_train, 
                cv=kf, scoring='neg_mean_squared_error'
            )
            
            # 转换为RMSE
            xgb_cv_rmse = np.sqrt(-xgb_cv_scores)
            
            # 线性回归交叉验证
            lr_cv_scores = cross_val_score(
                LinearRegression(), X_train_scaled, y_train, 
                cv=kf, scoring='neg_mean_squared_error'
            )
            
            # 转换为RMSE
            lr_cv_rmse = np.sqrt(-lr_cv_scores)
        
        # 显示交叉验证结果
        st.markdown("#### 交叉验证RMSE结果")
        
        cv_results = pd.DataFrame({
            'Fold': range(1, n_folds+1),
            'XGBoost': xgb_cv_rmse,
            'Linear Regression': lr_cv_rmse
        })
        
        st.table(cv_results.set_index('Fold').round(4))
        
        # 交叉验证结果可视化 - 使用新的绘图方式
        def plot_cv_results(ax):
            cv_results_melted = pd.melt(cv_results, id_vars=['Fold'], var_name='Model', value_name='RMSE')
            sns.barplot(x='Fold', y='RMSE', hue='Model', data=cv_results_melted, ax=ax)
        
        st.write(get_chinese_plot(
            plot_cv_results, 
            "交叉验证RMSE比较", 
            "折数", 
            "RMSE值",
            figsize=(10, 6)
        ), unsafe_allow_html=True)
        
        # 平均性能对比
        st.markdown("#### 平均交叉验证性能")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("XGBoost平均RMSE", f"{np.mean(xgb_cv_rmse):.4f}", f"{np.mean(lr_cv_rmse) - np.mean(xgb_cv_rmse):.4f}")
        with col2:
            st.metric("线性回归平均RMSE", f"{np.mean(lr_cv_rmse):.4f}")
        
        st.markdown("""
        交叉验证结果表明XGBoost模型的性能优于线性回归模型，且在不同数据分割上保持稳定。
        """)


    # 学习曲线
    with st.expander("学习曲线分析", expanded=False):
        st.markdown("### 学习曲线分析")
        
        st.markdown("""
        学习曲线可以帮助我们了解模型的过拟合/欠拟合状况，以及模型随着训练样本增加的性能变化。
        """)
        
        # 设置训练集比例
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        with st.echo():
            from sklearn.model_selection import learning_curve
            
            # 获取XGBoost学习曲线
            train_sizes_abs, train_scores, test_scores = learning_curve(
                xgb_model, X_train_scaled, y_train,
                train_sizes=train_sizes,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            # 转换为RMSE
            train_rmse = np.sqrt(-np.mean(train_scores, axis=1))
            test_rmse = np.sqrt(-np.mean(test_scores, axis=1))
        
        # 可视化学习曲线 - 使用新的绘图方式
        def plot_learning_curve(ax):
            ax.plot(train_sizes_abs, train_rmse, 'o-', label='训练集RMSE')
            ax.plot(train_sizes_abs, test_rmse, 'o-', label='验证集RMSE')
            ax.legend()
            ax.grid(True)
        
        st.write(get_chinese_plot(
            plot_learning_curve, 
            "XGBoost学习曲线", 
            "训练样本数", 
            "RMSE",
            figsize=(10, 6)
        ), unsafe_allow_html=True)
        
        # 学习曲线解释
        st.markdown("""
        **解读学习曲线:**
        
        - 如果训练误差和验证误差都很高，且两者之间差距较小，模型可能欠拟合
        - 如果训练误差很低但验证误差很高，模型可能过拟合
        - 理想情况下，随着样本数增加，训练误差和验证误差都应趋于接近并稳定在一个较低的值
        """)

    # 小实验：真实数据集上的应用
    with st.expander("真实数据集应用（选做）", expanded=False):
        st.markdown("### 真实数据集上的XGBoost应用")
        
        try:
            from sklearn.datasets import fetch_california_housing
            
            # 加载California房价数据集
            housing = fetch_california_housing()
            
            X_housing = housing.data
            y_housing = housing.target
            feature_names_housing = housing.feature_names
            
            st.markdown("""
            这里我们使用California房价数据集来测试XGBoost模型。该数据集包含8个特征和20640个样本。
            """)
            
            # 显示数据集信息
            st.markdown("#### 数据集概览")
            
            housing_df = pd.DataFrame(X_housing, columns=feature_names_housing)
            housing_df['PRICE'] = y_housing
            
            st.write(housing_df.head())
            
            # 数据集统计信息
            st.markdown("#### 统计信息")
            st.write(housing_df.describe())
            
            # 特征与目标变量的相关性 - 使用新的绘图方式
            st.markdown("#### 特征相关性")
            
            def plot_correlation(ax):
                corr = housing_df.corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            
            st.write(get_chinese_plot(
                plot_correlation, 
                "California房价数据集相关性矩阵", 
                figsize=(10, 8)
            ), unsafe_allow_html=True)
            
            # 训练测试分割
            X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
                X_housing, y_housing, test_size=0.2, random_state=42
            )
            
            # 标准化
            scaler_h = StandardScaler()
            X_train_h_scaled = scaler_h.fit_transform(X_train_h)
            X_test_h_scaled = scaler_h.transform(X_test_h)
            
            # 训练XGBoost模型
            if st.button("在California房价数据集上训练XGBoost"):
                with st.spinner("正在训练模型..."):
                    # 训练XGBoost模型
                    xgb_model_h = xgb.XGBRegressor(
                        objective='reg:squarederror',
                        max_depth=5,
                        learning_rate=0.1,
                        n_estimators=100,
                        random_state=42
                    )
                    
                    xgb_model_h.fit(X_train_h_scaled, y_train_h)
                    
                    # 预测
                    xgb_pred_h = xgb_model_h.predict(X_test_h_scaled)
                    
                    # 评估
                    xgb_rmse_h = np.sqrt(mean_squared_error(y_test_h, xgb_pred_h))
                    xgb_r2_h = r2_score(y_test_h, xgb_pred_h)
                    
                    st.success("模型训练完成！")
                    
                    # 显示结果
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("RMSE", f"{xgb_rmse_h:.4f}")
                    with col2:
                        st.metric("R²", f"{xgb_r2_h:.4f}")
                    
                    # 预测vs实际 - 使用新的绘图方式
                    def plot_pred_actual(ax):
                        ax.scatter(y_test_h, xgb_pred_h, alpha=0.5)
                        ax.plot([y_test_h.min(), y_test_h.max()], [y_test_h.min(), y_test_h.max()], 'r--')
                    
                    st.write(get_chinese_plot(
                        plot_pred_actual, 
                        "XGBoost在California房价数据集上的预测效果", 
                        "实际房价", 
                        "预测房价",
                        figsize=(10, 6)
                    ), unsafe_allow_html=True)
                    
                    # 特征重要性 - 使用新的绘图方式
                    def plot_importance(ax):
                        xgb.plot_importance(xgb_model_h, ax=ax)
                    
                    st.write(get_chinese_plot(
                        plot_importance, 
                        "特征重要性", 
                        figsize=(10, 6)
                    ), unsafe_allow_html=True)
        except:
            st.warning("无法加载California房价数据集。请确保已安装scikit-learn并且有网络连接。") 