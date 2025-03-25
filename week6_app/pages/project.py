import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# 导入matplotlib中文字体支持
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.plot_utils import configure_matplotlib_fonts

# 配置matplotlib支持中文
configure_matplotlib_fonts()

def show():
    st.title("小组项目二：房价预测模型优化 (XGBoost)")
    
    st.markdown("""
    本页面提供房价预测模型优化项目的指导和资源，帮助您应用所学的XGBoost知识解决实际问题。
    """)
    
    # 项目介绍
    st.header("项目介绍")
    st.markdown("""
    ### 项目目标
    
    在上周线性/多项式回归模型的基础上，使用XGBoost算法优化房价预测模型，提高预测性能。
    
    ### 项目要求
    
    * 进行模型参数调优，提高预测性能
    * 比较XGBoost与线性回归、多项式回归的性能差异
    * 分析特征重要性，解释模型预测结果
    
    ### 提交内容
    
    1. 代码实现
       * XGBoost模型的实现代码
       * 参数调优策略与实现
       * 性能对比代码
    
    2. 实验报告
       * 数据集描述
       * 预处理步骤
       * 不同模型的性能指标比较
       * 参数调优过程与结果
       * 特征重要性分析
       * 结论与讨论
    """)
    
    # 数据集介绍
    st.header("数据集")
    st.markdown("""
    本项目使用房价预测数据集，包含各种住房特征和对应的销售价格。你可以选择使用：
    
    1. 经典的波士顿房价数据集
    2. Kaggle上的Ames房价数据集
    3. 其他任何合适的房价相关数据集
    
    下面提供了一个示例数据预览：
    """)
    
    # 生成示例数据
    try:
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing()
        
        # 创建DataFrame
        housing_df = pd.DataFrame(
            housing.data, 
            columns=housing.feature_names
        )
        housing_df['PRICE'] = housing.target
        
        # 显示数据集
        st.dataframe(housing_df.head())
        
        # 数据集描述
        st.markdown("### 数据集描述")
        st.markdown(f"""
        * 数据集名称：California Housing
        * 样本数量：{housing.data.shape[0]}
        * 特征数量：{housing.data.shape[1]}
        * 目标变量：房屋价格中位数
        """)
        
        # 数据可视化
        st.markdown("### 数据预览")
        
        # 目标变量分布
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(housing_df['PRICE'], kde=True, ax=ax)
        ax.set_title('房价分布')
        ax.set_xlabel('价格')
        ax.set_ylabel('频率')
        st.pyplot(fig)
        
        # 相关性热图
        st.markdown("### 特征相关性")
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = housing_df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title('特征相关性热图')
        st.pyplot(fig)
        
    except:
        st.warning("无法加载示例数据集。这不会影响您的项目，您可以使用自己选择的数据集。")
        
        # 显示虚拟数据
        st.markdown("示例数据格式：")
        
        # 创建虚拟数据
        np.random.seed(42)
        n_samples = 5
        virtual_df = pd.DataFrame({
            'SquareFootage': np.random.randint(1000, 3000, n_samples),
            'Bedrooms': np.random.randint(2, 6, n_samples),
            'Bathrooms': np.random.randint(1, 4, n_samples),
            'YearBuilt': np.random.randint(1950, 2020, n_samples),
            'Neighborhood': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples),
            'Price': np.random.randint(200000, 800000, n_samples)
        })
        
        st.dataframe(virtual_df)
    
    # 项目实施指南
    st.header("项目实施指南")
    
    steps = {
        "数据准备与特征工程": {
            "content": """
            1. 加载数据集并进行基本探索
            2. 处理缺失值和异常值
            3. 对类别特征进行编码
            4. 进行特征选择和特征变换
            5. 划分训练集和测试集
            """,
            "code": """
            import pandas as pd
            import numpy as np
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            
            # 加载数据
            df = pd.read_csv('房价数据集.csv')
            
            # 查看基本信息
            print(df.info())
            print(df.describe())
            
            # 检查缺失值
            print(df.isnull().sum())
            
            # 处理缺失值 (示例)
            df = df.fillna(df.mean())  # 数值型特征用均值填充
            
            # 分离特征和目标
            X = df.drop('price', axis=1)
            y = df['price']
            
            # 识别数值和类别特征
            numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = X.select_dtypes(include=['object']).columns
            
            # 创建预处理管道
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ])
            
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            """
        },
        "基线模型创建": {
            "content": """
            1. 实现线性回归模型
            2. 实现多项式回归模型
            3. 评估基线模型性能
            """,
            "code": """
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.metrics import mean_squared_error, r2_score
            import numpy as np
            
            # 线性回归模型
            linear_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', LinearRegression())
            ])
            
            linear_pipeline.fit(X_train, y_train)
            linear_pred = linear_pipeline.predict(X_test)
            
            # 多项式回归模型
            poly_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('poly', PolynomialFeatures(degree=2)),
                ('regressor', LinearRegression())
            ])
            
            poly_pipeline.fit(X_train, y_train)
            poly_pred = poly_pipeline.predict(X_test)
            
            # 评估性能
            def evaluate_model(y_true, y_pred, model_name):
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)
                print(f"{model_name} - RMSE: {rmse:.2f}, R²: {r2:.4f}")
                return rmse, r2
            
            linear_rmse, linear_r2 = evaluate_model(y_test, linear_pred, "线性回归")
            poly_rmse, poly_r2 = evaluate_model(y_test, poly_pred, "多项式回归")
            """
        },
        "XGBoost模型开发与调优": {
            "content": """
            1. 创建基础XGBoost模型
            2. 使用交叉验证评估初始性能
            3. 实施参数调优
            4. 评估最终模型性能
            """,
            "code": """
            import xgboost as xgb
            from sklearn.model_selection import GridSearchCV, cross_val_score
            
            # 创建XGBoost模型管道
            xgb_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('xgb', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))
            ])
            
            # 初始模型评估
            xgb_pipeline.fit(X_train, y_train)
            xgb_pred = xgb_pipeline.predict(X_test)
            xgb_rmse, xgb_r2 = evaluate_model(y_test, xgb_pred, "基础XGBoost")
            
            # 参数网格
            param_grid = {
                'xgb__max_depth': [3, 5, 7],
                'xgb__learning_rate': [0.01, 0.1, 0.2],
                'xgb__n_estimators': [100, 200],
                'xgb__min_child_weight': [1, 3, 5],
                'xgb__subsample': [0.7, 0.8, 0.9],
                'xgb__colsample_bytree': [0.7, 0.8, 0.9]
            }
            
            # 网格搜索
            grid_search = GridSearchCV(
                xgb_pipeline,
                param_grid,
                cv=5,
                scoring='neg_mean_squared_error',
                verbose=1,
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # 最优模型
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            # 评估最优模型
            best_pred = best_model.predict(X_test)
            best_rmse, best_r2 = evaluate_model(y_test, best_pred, "调优后XGBoost")
            """
        },
        "结果分析与可视化": {
            "content": """
            1. 比较不同模型的性能
            2. 分析XGBoost特征重要性
            3. 可视化预测结果
            4. 撰写项目报告
            """,
            "code": """
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 模型性能比较
            models = ['线性回归', '多项式回归', '基础XGBoost', '调优XGBoost']
            rmse_values = [linear_rmse, poly_rmse, xgb_rmse, best_rmse]
            r2_values = [linear_r2, poly_r2, xgb_r2, best_r2]
            
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.bar(models, rmse_values)
            plt.title('RMSE比较')
            plt.ylabel('RMSE (越低越好)')
            
            plt.subplot(1, 2, 2)
            plt.bar(models, r2_values)
            plt.title('R²比较')
            plt.ylabel('R² (越高越好)')
            
            plt.tight_layout()
            plt.savefig('model_comparison.png')
            
            # 特征重要性分析
            xgb_model = best_model.named_steps['xgb']
            
            plt.figure(figsize=(10, 6))
            xgb.plot_importance(xgb_model, max_num_features=10)
            plt.title('XGBoost特征重要性')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            
            # 预测vs实际值
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, best_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.xlabel('实际值')
            plt.ylabel('预测值')
            plt.title('XGBoost预测vs实际值')
            plt.tight_layout()
            plt.savefig('prediction_vs_actual.png')
            """
        }
    }
    
    # 显示步骤
    selected_step = st.selectbox("选择查看步骤", list(steps.keys()))
    
    st.subheader(selected_step)
    st.markdown(steps[selected_step]["content"])
    
    with st.expander("示例代码"):
        st.code(steps[selected_step]["code"], language="python")
    
    # 评估标准
    st.header("评估标准")
    st.markdown("""
    项目评估将基于以下几个方面：
    
    1. **技术实现 (40%)**
       * XGBoost模型实现的正确性
       * 参数调优策略的合理性
       * 代码质量和可读性
    
    2. **性能提升 (30%)**
       * 相比基线模型的性能提升
       * 评估指标的选择合理性
    
    3. **分析深度 (20%)**
       * 特征重要性分析的质量
       * 对模型行为的解释
       * 对结果的讨论深度
    
    4. **报告质量 (10%)**
       * 报告结构和逻辑性
       * 可视化效果
       * 表达清晰度
    """)
    
    # 资源链接
    st.header("参考资源")
    st.markdown("""
    * [XGBoost官方文档](https://xgboost.readthedocs.io/)
    * [Scikit-learn文档](https://scikit-learn.org/stable/)
    * [特征工程指南](https://www.kaggle.com/learn/feature-engineering)
    * [参数调优最佳实践](https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning)
    """)
    
    # 常见问题
    with st.expander("常见问题解答"):
        st.markdown("""
        **Q: 数据预处理需要做到什么程度？**  
        A: 应该至少处理缺失值、异常值，并对类别特征进行适当编码。根据数据特点，可能还需要进行特征缩放和特征工程。
        
        **Q: 必须使用网格搜索进行参数调优吗？**  
        A: 不一定。可以使用网格搜索、随机搜索、贝叶斯优化等方法，或者手动调优也可以。关键是详细记录调优过程和结果。
        
        **Q: 如何评估特征重要性？**  
        A: XGBoost内置了特征重要性评估功能，可以使用`plot_importance()`函数。此外，也可以使用SHAP值等方法进行更深入的分析。
        
        **Q: 报告需要多长？**  
        A: 没有严格的长度要求，但应该包含所有必要的内容。通常5-10页足够。质量比数量更重要。
        """)
    
    # 提交指南
    st.header("提交指南")
    st.info("""
    1. 将代码和报告打包成一个ZIP文件
    2. 命名格式：小组号_XGBoost房价预测
    3. 在课程平台上提交
    4. 截止日期：第7周课前
    """)
    
    # 帮助支持
    st.header("帮助与支持")
    st.markdown("""
    如果您在项目实施过程中遇到任何问题，请通过以下渠道寻求帮助：
    
    * 课程论坛的项目讨论区
    * 课后答疑时间
    * 向助教发送邮件
    """)
    
    # 激励信息
    st.success("祝您项目顺利！通过这个项目，您将深入理解XGBoost算法并掌握其在实际问题中的应用。") 