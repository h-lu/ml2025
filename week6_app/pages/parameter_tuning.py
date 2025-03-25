import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import time
import sys
import os

# 导入工具函数
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_utils import generate_complex_data, split_and_preprocess_data
from utils.model_utils import train_xgboost_model, evaluate_model, plot_feature_importance
# 导入matplotlib中文字体支持
from utils.plot_utils import configure_matplotlib_fonts

# 配置matplotlib支持中文
configure_matplotlib_fonts()

def show():
    st.title("XGBoost参数调优实验")
    
    # 生成示例数据
    st.markdown("### 数据准备")
    st.markdown("""
    首先，我们将生成一个包含非线性关系的数据集来测试XGBoost参数调优效果。
    这个数据集具有多种非线性模式，适合展示XGBoost的优势。
    """)
    
    # 用户控制样本数
    n_samples = st.slider("样本数量", 500, 3000, 1000, step=100)
    
    with st.echo():
        # 生成数据
        np.random.seed(42)
        X = np.random.rand(n_samples, 6)  # 6个特征
        # 创建复杂的非线性关系
        y = (
            2 + 
            3 * X[:, 0]**2 + 
            2 * np.sin(2 * np.pi * X[:, 1]) + 
            4 * (X[:, 2] - 0.5)**2 + 
            X[:, 3] * X[:, 4] +  # 交互特征
            1.5 * np.exp(X[:, 5]) + 
            np.random.normal(0, 0.5, n_samples)  # 噪声
        )
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
    # 显示数据集信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("训练样本数", len(X_train))
    with col2:
        st.metric("测试样本数", len(X_test))
    with col3:
        st.metric("特征数量", X.shape[1])
    
    # 基础模型
    st.markdown("### 基础XGBoost模型")
    st.markdown("""
    我们先创建一个基础的XGBoost模型作为参考点。然后通过调整参数来观察性能变化。
    """)
    
    with st.echo():
        # 创建基础XGBoost模型
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        # 训练模型
        base_model.fit(X_train_scaled, y_train)
        
        # 预测
        y_pred_base = base_model.predict(X_test_scaled)
        
        # 评估
        rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
        r2_base = r2_score(y_test, y_pred_base)
    
    # 显示基础模型性能
    col1, col2 = st.columns(2)
    with col1:
        st.metric("基础模型 RMSE", f"{rmse_base:.4f}")
    with col2:
        st.metric("基础模型 R²", f"{r2_base:.4f}")
    
    # 单参数影响实验
    st.markdown("### 单参数影响分析")
    st.markdown("""
    下面我们将研究不同参数对模型性能的影响。通过调整单个参数并保持其他参数不变，
    我们可以观察到每个参数的单独效果。
    """)
    
    param_choice = st.selectbox(
        "选择要研究的参数",
        ["max_depth", "learning_rate", "n_estimators", "min_child_weight", "subsample", "colsample_bytree", "gamma", "reg_alpha", "reg_lambda"]
    )
    
    # 参数范围设置
    param_ranges = {
        "max_depth": list(range(1, 11)),
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
        "n_estimators": [10, 50, 100, 200, 300, 500],
        "min_child_weight": [1, 3, 5, 7, 10],
        "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2, 0.5, 1.0, 2.0],
        "reg_alpha": [0, 0.1, 0.5, 1.0, 2.0, 5.0],
        "reg_lambda": [0, 0.1, 0.5, 1.0, 2.0, 5.0]
    }
    
    # 参数说明
    param_descriptions = {
        "max_depth": "树的最大深度。增加该值可以使模型更复杂，但过大可能导致过拟合。",
        "learning_rate": "学习率。较小的值需要更多的树，但可以获得更好的性能。",
        "n_estimators": "树的数量。增加树的数量可以改善性能，但会增加计算成本。",
        "min_child_weight": "子节点中所需的最小样本权重和。较大的值可以防止过拟合。",
        "subsample": "用于训练每棵树的样本比例。小于1可以防止过拟合。",
        "colsample_bytree": "构建每棵树时考虑的特征比例。小于1可以防止过拟合。",
        "gamma": "节点分裂所需的最小损失减少值。值越大，算法越保守。",
        "reg_alpha": "L1正则化项。有助于减少模型复杂度，处理稀疏特征。",
        "reg_lambda": "L2正则化项。有助于减少模型复杂度，一般来说所有场景都适用。"
    }
    
    # 显示参数说明
    st.info(param_descriptions[param_choice])
    
    # 运行参数实验
    if st.button(f"运行 {param_choice} 参数实验"):
        with st.spinner("正在进行参数实验..."):
            # 准备存储结果
            results = []
            
            # 默认参数
            default_params = {
                'objective': 'reg:squarederror',
                'max_depth': 5,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'random_state': 42
            }
            
            # 创建进度条
            progress_bar = st.progress(0)
            
            # 测试不同参数值
            for i, value in enumerate(param_ranges[param_choice]):
                # 更新参数
                current_params = default_params.copy()
                current_params[param_choice] = value
                
                # 创建并训练模型
                model = xgb.XGBRegressor(**current_params)
                model.fit(X_train_scaled, y_train)
                
                # 预测并评估
                y_pred = model.predict(X_test_scaled)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                # 存储结果
                results.append({
                    'param_value': value,
                    'rmse': rmse,
                    'r2': r2
                })
                
                # 更新进度条
                progress_bar.progress((i + 1) / len(param_ranges[param_choice]))
            
            # 转换为DataFrame
            results_df = pd.DataFrame(results)
            
            # 创建图表
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # RMSE图
            sns.lineplot(x='param_value', y='rmse', data=results_df, marker='o', ax=axes[0])
            axes[0].set_title(f'{param_choice} vs RMSE')
            axes[0].set_xlabel(param_choice)
            axes[0].set_ylabel('RMSE (越低越好)')
            axes[0].grid(True)
            
            # R²图
            sns.lineplot(x='param_value', y='r2', data=results_df, marker='o', ax=axes[1])
            axes[1].set_title(f'{param_choice} vs R²')
            axes[1].set_xlabel(param_choice)
            axes[1].set_ylabel('R² (越高越好)')
            axes[1].grid(True)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # 显示结果表格
            st.markdown("#### 详细结果")
            st.table(results_df.set_index('param_value').round(4))
            
            # 找出最佳参数
            best_rmse_idx = results_df['rmse'].idxmin()
            best_r2_idx = results_df['r2'].idxmax()
            
            best_param_rmse = results_df.loc[best_rmse_idx, 'param_value']
            best_param_r2 = results_df.loc[best_r2_idx, 'param_value']
            
            # 显示最佳参数
            st.markdown("#### 最佳参数值")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"RMSE最低的 {param_choice} 值", best_param_rmse)
            with col2:
                st.metric(f"R²最高的 {param_choice} 值", best_param_r2)
    
    # 多参数网格搜索
    st.markdown("### 多参数网格搜索")
    st.markdown("""
    上面我们只分析了单个参数的影响，但在实际应用中，参数之间可能存在交互作用。
    下面我们将使用网格搜索同时调整多个参数。
    
    注意：网格搜索可能需要较长时间运行，尤其是参数空间较大时。
    """)
    
    # 选择要优化的参数
    st.markdown("#### 选择要优化的参数")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_max_depth = st.checkbox("优化 max_depth", value=True)
        use_learning_rate = st.checkbox("优化 learning_rate", value=True)
        use_n_estimators = st.checkbox("优化 n_estimators", value=False)
        use_min_child_weight = st.checkbox("优化 min_child_weight", value=False)
        use_gamma = st.checkbox("优化 gamma", value=False)
    
    with col2:
        use_subsample = st.checkbox("优化 subsample", value=False)
        use_colsample_bytree = st.checkbox("优化 colsample_bytree", value=False)
        use_reg_alpha = st.checkbox("优化 reg_alpha", value=False)
        use_reg_lambda = st.checkbox("优化 reg_lambda", value=False)
    
    # 设置参数范围
    st.markdown("#### 设置参数搜索范围")
    param_grid = {}
    
    if use_max_depth:
        param_grid['max_depth'] = st.multiselect(
            "max_depth 值", 
            options=[1, 3, 5, 7, 9],
            default=[3, 5, 7]
        )
    
    if use_learning_rate:
        param_grid['learning_rate'] = st.multiselect(
            "learning_rate 值", 
            options=[0.01, 0.05, 0.1, 0.2],
            default=[0.05, 0.1]
        )
    
    if use_n_estimators:
        param_grid['n_estimators'] = st.multiselect(
            "n_estimators 值", 
            options=[50, 100, 200, 300],
            default=[100, 200]
        )
    
    if use_min_child_weight:
        param_grid['min_child_weight'] = st.multiselect(
            "min_child_weight 值", 
            options=[1, 3, 5, 7],
            default=[1, 3]
        )
    
    if use_subsample:
        param_grid['subsample'] = st.multiselect(
            "subsample 值", 
            options=[0.6, 0.7, 0.8, 0.9, 1.0],
            default=[0.8, 0.9]
        )
    
    if use_colsample_bytree:
        param_grid['colsample_bytree'] = st.multiselect(
            "colsample_bytree 值", 
            options=[0.6, 0.7, 0.8, 0.9, 1.0],
            default=[0.8, 0.9]
        )
    
    if use_gamma:
        param_grid['gamma'] = st.multiselect(
            "gamma 值", 
            options=[0, 0.1, 0.3, 0.5, 1.0],
            default=[0, 0.1]
        )
    
    if use_reg_alpha:
        param_grid['reg_alpha'] = st.multiselect(
            "reg_alpha 值", 
            options=[0, 0.1, 0.5, 1.0],
            default=[0, 0.1]
        )
    
    if use_reg_lambda:
        param_grid['reg_lambda'] = st.multiselect(
            "reg_lambda 值", 
            options=[0.1, 0.5, 1.0, 2.0],
            default=[0.5, 1.0]
        )
    
    # 计算组合数
    combinations = 1
    for param in param_grid.values():
        combinations *= len(param)
    
    # 显示组合数
    st.metric("参数组合总数", combinations)
    
    # 设置交叉验证折数
    cv_folds = st.slider("交叉验证折数", 2, 10, 3)
    
    # 运行网格搜索
    if st.button("运行网格搜索") and combinations > 0:
        if combinations > 100:
            warning = st.warning(f"注意：参数组合数较大 ({combinations})，可能需要较长时间。")
            continue_anyway = st.button("仍然继续")
            if not continue_anyway:
                st.stop()
            warning.empty()
        
        start_time = time.time()
        with st.spinner(f"正在进行网格搜索 ({combinations}个组合)..."):
            # 创建基础XGBoost模型
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42
            )
            
            # 创建网格搜索
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring='neg_mean_squared_error',
                cv=cv_folds,
                verbose=1,
                n_jobs=-1
            )
            
            # 执行网格搜索
            grid_search.fit(X_train_scaled, y_train)
            
            # 获取最佳参数
            best_params = grid_search.best_params_
            
            # 用最佳参数创建模型
            best_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                **best_params
            )
            
            # 训练最佳模型
            best_model.fit(X_train_scaled, y_train)
            
            # 预测并评估
            y_pred_best = best_model.predict(X_test_scaled)
            rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
            r2_best = r2_score(y_test, y_pred_best)
        
        # 计算运行时间
        elapsed_time = time.time() - start_time
        
        # 显示最佳参数
        st.markdown("#### 最佳参数组合")
        st.json(best_params)
        
        # 显示性能对比
        st.markdown("#### 性能对比")
        
        improvement_rmse = rmse_base - rmse_best
        improvement_r2 = r2_best - r2_base
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("调优后 RMSE", f"{rmse_best:.4f}", f"{improvement_rmse:.4f}")
        with col2:
            st.metric("调优后 R²", f"{r2_best:.4f}", f"{improvement_r2:.4f}")
        with col3:
            st.metric("网格搜索耗时", f"{elapsed_time:.2f}秒")
        
        # 预测vs实际值比较
        st.markdown("#### 调优前后模型性能比较")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 基础模型
        axes[0].scatter(y_test, y_pred_base, alpha=0.5)
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        axes[0].set_xlabel('实际值')
        axes[0].set_ylabel('预测值')
        axes[0].set_title('基础模型: 预测 vs 实际')
        
        # 调优后模型
        axes[1].scatter(y_test, y_pred_best, alpha=0.5)
        axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        axes[1].set_xlabel('实际值')
        axes[1].set_ylabel('预测值')
        axes[1].set_title('调优后模型: 预测 vs 实际')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 显示特征重要性
        st.markdown("#### 调优后模型的特征重要性")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        xgb.plot_importance(best_model, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        
        # 决策树可视化（可选）
        if st.checkbox("显示第一棵决策树可视化"):
            st.markdown("#### 第一棵决策树的可视化")
            
            # 保存为图片文件
            xgb.plot_tree(best_model, num_trees=0)
            plt.title("第一棵决策树")
            plt.tight_layout()
            fig = plt.gcf()
            fig.set_size_inches(15, 10)
            st.pyplot(fig)
    
    # 参数调优建议
    st.markdown("### 参数调优最佳实践")
    
    st.markdown("""
    #### XGBoost参数调优策略
    
    1. **先调整模型复杂度参数**:
       * `max_depth`: 通常从3-6开始尝试
       * `min_child_weight`: 可以从1开始，逐步增加
    
    2. **然后调整随机性参数**:
       * `subsample`: 通常在0.7-1.0之间
       * `colsample_bytree`: 通常在0.7-1.0之间
    
    3. **接着调整正则化参数**:
       * `gamma`: 根据任务复杂性，通常从0开始
       * `reg_alpha`和`reg_lambda`: 可以从0和1开始
    
    4. **最后调整学习率和树的数量**:
       * 降低`learning_rate`(通常≤0.1)
       * 增加`n_estimators`(通常增加到几百或更多)
    
    5. **注意事项**:
       * 使用早停(early stopping)避免过拟合
       * 对于大型数据集，可以从较小的参数网格开始
       * 记录每次调整的结果，以理解参数的影响
    """)
    
    st.info("""
    记住：参数调优是一个反复试验的过程，不同的数据集可能需要不同的参数设置。
    始终保持对模型性能的监控，确保调优带来实质性的改进。
    """)

    # 交互练习  
    with st.expander("参数调优小练习", expanded=False):
        st.markdown("### 交互式参数调优练习")
        
        st.markdown("""
        现在您已经了解了不同参数的作用，请尝试手动调整下面的参数，看能否提高模型性能。
        目标是通过调整参数使RMSE最小化。
        """)
        
        # 用户可调参数
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_depth_user = st.slider("max_depth", 1, 10, 5)
            learning_rate_user = st.slider("learning_rate", 0.01, 0.5, 0.1, step=0.01)
            n_estimators_user = st.slider("n_estimators", 50, 500, 100, step=10)
        
        with col2:
            min_child_weight_user = st.slider("min_child_weight", 1, 10, 1)
            subsample_user = st.slider("subsample", 0.5, 1.0, 0.8, step=0.05)
            colsample_bytree_user = st.slider("colsample_bytree", 0.5, 1.0, 0.8, step=0.05)
        
        with col3:
            gamma_user = st.slider("gamma", 0.0, 5.0, 0.0, step=0.1)
            reg_alpha_user = st.slider("reg_alpha", 0.0, 5.0, 0.0, step=0.1)
            reg_lambda_user = st.slider("reg_lambda", 0.1, 5.0, 1.0, step=0.1)
        
        # 训练按钮
        if st.button("训练自定义模型"):
            with st.spinner("训练中..."):
                # 创建用户定义的模型
                user_model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    max_depth=max_depth_user,
                    learning_rate=learning_rate_user,
                    n_estimators=n_estimators_user,
                    min_child_weight=min_child_weight_user,
                    subsample=subsample_user,
                    colsample_bytree=colsample_bytree_user,
                    gamma=gamma_user,
                    reg_alpha=reg_alpha_user,
                    reg_lambda=reg_lambda_user,
                    random_state=42
                )
                
                # 训练模型
                user_model.fit(X_train_scaled, y_train)
                
                # 预测并评估
                y_pred_user = user_model.predict(X_test_scaled)
                rmse_user = np.sqrt(mean_squared_error(y_test, y_pred_user))
                r2_user = r2_score(y_test, y_pred_user)
                
                # 与基础模型比较
                improvement_rmse = rmse_base - rmse_user
                improvement_r2 = r2_user - r2_base
                
                # 显示结果
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("您的模型 RMSE", f"{rmse_user:.4f}", f"{improvement_rmse:.4f}")
                with col2:
                    st.metric("您的模型 R²", f"{r2_user:.4f}", f"{improvement_r2:.4f}")
                
                # 评价表现
                if improvement_rmse > 0:
                    st.success(f"恭喜！您的模型比基础模型表现好 {improvement_rmse:.4f} RMSE。")
                    
                    if improvement_rmse > 0.2:
                        st.balloons()
                        st.success("出色的调优！您的参数设置非常有效。")
                    elif improvement_rmse > 0.1:
                        st.success("很好的调优！还有进一步改进的空间。")
                    else:
                        st.info("参数调整产生了积极影响，但效果有限。尝试更大幅度的调整？")
                else:
                    st.warning("您的模型表现不如基础模型。请尝试不同的参数组合。")
                    st.info("提示：尝试减小max_depth或增加正则化参数可能有助于提高性能。") 