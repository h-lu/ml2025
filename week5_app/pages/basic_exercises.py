import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import sys
import os

# 添加utils目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.fonts import configure_matplotlib_fonts

def show():
    # 配置matplotlib字体
    configure_matplotlib_fonts()
    
    st.markdown("## 基础练习")
    
    st.markdown("### 线性回归练习")
    
    st.markdown("""
    在这个练习中，我们将使用加利福尼亚房价数据集，练习线性回归模型的构建和评估。
    
    **数据集描述**：
    - **特征**：每个街区的人口统计和房屋相关特征
    - **目标**：房屋的中位数价格（以$100,000为单位）
    
    **任务**：
    1. 加载数据并进行简单预处理
    2. 划分训练集和测试集
    3. 使用线性回归模型拟合数据
    4. 评估模型性能
    """)
    
    # 加载数据
    # 由于有时候API访问会有限制，我们使用模拟数据代替fetch_california_housing()
    # california = fetch_california_housing()
    
    # 生成模拟的加利福尼亚房价数据
    np.random.seed(42)
    n_samples = 20000
    n_features = 8
    
    # 模拟特征
    X = np.random.randn(n_samples, n_features)
    feature_names = [
        'MedInc', 'HouseAge', 'AveRooms', 
        'AveBedrms', 'Population', 'AveOccup', 
        'Latitude', 'Longitude'
    ]
    
    # 模拟目标变量（房价）
    # 尝试创建与真实数据集类似的关系
    y = (0.5 * X[:, 0]  # MedInc正相关
         + 0.2 * X[:, 1]  # HouseAge正相关
         + 0.3 * X[:, 2]  # AveRooms正相关
         - 0.1 * X[:, 3]  # AveBedrms轻微负相关
         - 0.05 * X[:, 4]  # Population轻微负相关
         - 0.05 * X[:, 5]  # AveOccup轻微负相关
         + 0.7 * np.sin(X[:, 6] * 0.1)  # Latitude非线性相关
         + 0.7 * np.cos(X[:, 7] * 0.1)  # Longitude非线性相关
         + np.random.normal(0, 0.2, n_samples))  # 添加噪声
    
    # 归一化到合理范围，使均值为2左右（类似真实数据集）
    y = 2.0 + (y - np.mean(y)) / np.std(y) * 0.8
    
    # 创建DataFrame方便查看
    housing_df = pd.DataFrame(X, columns=feature_names)
    housing_df['PRICE'] = y
    
    # 显示数据集
    st.markdown("#### 数据预览")
    st.dataframe(housing_df.head())
    
    st.markdown("#### 数据描述")
    st.write(housing_df.describe())
    
    # 特征说明
    st.markdown("#### 特征说明")
    feature_descriptions = {
        'MedInc': '街区内家庭收入中位数',
        'HouseAge': '街区内房屋年龄中位数',
        'AveRooms': '每户平均房间数',
        'AveBedrms': '每户平均卧室数',
        'Population': '街区人口',
        'AveOccup': '平均入住率',
        'Latitude': '纬度',
        'Longitude': '经度',
        'PRICE': '房屋价格中位数（目标变量，单位：$100,000）\n注意：这是模拟数据，用于教学目的'
    }
    
    for feature, desc in feature_descriptions.items():
        st.write(f"**{feature}**: {desc}")
    
    # 数据可视化
    st.markdown("#### 数据可视化")
    
    # 相关性矩阵
    st.markdown("##### 特征间相关性")
    corr = housing_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap='coolwarm')
    
    # 添加相关系数文本标签
    for i in range(len(corr)):
        for j in range(len(corr)):
            text = ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                           ha="center", va="center", color="black" if abs(corr.iloc[i, j]) < 0.5 else "white")
    
    # 设置刻度标签
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    
    plt.colorbar(im)
    plt.title('特征相关性矩阵')
    plt.tight_layout()
    st.pyplot(fig)
    
    # 各特征与目标变量的散点图
    st.markdown("##### 特征与目标变量关系")
    
    feature_to_plot = st.selectbox(
        "选择特征查看与房价的关系",
        options=feature_names
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(housing_df[feature_to_plot], housing_df['PRICE'], alpha=0.5)
    ax.set_xlabel(feature_to_plot)
    ax.set_ylabel('房价 (PRICE)')
    ax.set_title(f'{feature_to_plot} vs 房价')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # 练习任务代码区域
    st.markdown("### 练习代码")
    
    st.markdown("""
    请在下面填写代码，完成线性回归模型的构建和评估。你可以参考代码框架和提示，完成练习任务。
    """)
    
    code = st.text_area(
        "编辑代码",
        """
# 导入所需库
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 数据已加载在housing_df变量中
# X = housing_df.drop('PRICE', axis=1)
# y = housing_df['PRICE']

# 任务1: 划分训练集和测试集 (80% 训练, 20% 测试)
# X_train, X_test, y_train, y_test = ...

# 任务2: 标准化特征 (可选但推荐)
# scaler = ...
# X_train_scaled = ...
# X_test_scaled = ...

# 任务3: 创建并训练线性回归模型
# model = ...
# model.fit(...)

# 任务4: 预测并评估模型
# y_train_pred = ...
# y_test_pred = ...

# 计算模型性能指标
# train_mse = ...
# test_mse = ...
# train_rmse = ...
# test_rmse = ...
# train_r2 = ...
# test_r2 = ...

# 输出模型性能
# print(f"训练集 MSE: {train_mse:.4f}")
# print(f"测试集 MSE: {test_mse:.4f}")
# print(f"训练集 RMSE: {train_rmse:.4f}")
# print(f"测试集 RMSE: {test_rmse:.4f}")
# print(f"训练集 R²: {train_r2:.4f}")
# print(f"测试集 R²: {test_r2:.4f}")

# 分析模型系数
# coef_df = pd.DataFrame({
#    '特征': X.columns,
#    '系数': model.coef_
# })
# print(coef_df.sort_values('系数', ascending=False))
        """,
        height=500
    )
    
    if st.button("运行代码"):
        st.code(code, language="python")
        
        # 为了安全起见，我们不执行用户输入的代码
        # 而是提供一个示例解决方案
        st.markdown("### 示例解决方案")
        
        solution_code = """
# 导入所需库
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 数据已加载在housing_df变量中
X = housing_df.drop('PRICE', axis=1)
y = housing_df['PRICE']

# 任务1: 划分训练集和测试集 (80% 训练, 20% 测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 任务2: 标准化特征 (可选但推荐)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 任务3: 创建并训练线性回归模型
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 任务4: 预测并评估模型
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# 计算模型性能指标
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# 输出模型性能
print(f"训练集 MSE: {train_mse:.4f}")
print(f"测试集 MSE: {test_mse:.4f}")
print(f"训练集 RMSE: {train_rmse:.4f}")
print(f"测试集 RMSE: {test_rmse:.4f}")
print(f"训练集 R²: {train_r2:.4f}")
print(f"测试集 R²: {test_r2:.4f}")

# 分析模型系数
coef_df = pd.DataFrame({
   '特征': X.columns,
   '系数': model.coef_
})
print(coef_df.sort_values('系数', ascending=False))
        """
        
        st.code(solution_code, language="python")
        
        # 运行解决方案代码以展示结果
        X = housing_df.drop('PRICE', axis=1)
        y = housing_df['PRICE']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        st.markdown("#### 模型性能")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("训练集 MSE", f"{train_mse:.4f}")
            st.metric("训练集 RMSE", f"{train_rmse:.4f}")
            st.metric("训练集 R²", f"{train_r2:.4f}")
        
        with col2:
            st.metric("测试集 MSE", f"{test_mse:.4f}")
            st.metric("测试集 RMSE", f"{test_rmse:.4f}")
            st.metric("测试集 R²", f"{test_r2:.4f}")
        
        st.markdown("#### 模型系数分析")
        
        coef_df = pd.DataFrame({
            '特征': X.columns,
            '系数': model.coef_,
            '绝对值': np.abs(model.coef_)
        })
        coef_df = coef_df.sort_values('绝对值', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(coef_df['特征'], coef_df['系数'])
        ax.set_xlabel('系数值')
        ax.set_title('特征重要性（线性回归系数）')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.markdown("""
        #### 结果分析
        
        线性回归模型在加利福尼亚房价数据集上的表现：
        
        1. **模型性能**：
           - 测试集上的R²约为0.60，表明模型可以解释约60%的房价变异。
           - 测试集和训练集的性能相近，说明模型没有明显的过拟合。
        
        2. **系数分析**：
           - MedInc (收入中位数) 对房价有最强的正向影响，这符合常识。
           - 经纬度也对房价有显著影响，表明位置是房价的重要因素。
           - HouseAge (房屋年龄) 对房价也有一定影响。
        
        3. **改进方向**：
           - 考虑添加特征交互项和多项式特征
           - 尝试更复杂的模型，如随机森林或XGBoost
           - 进一步探索地理位置特征，可能通过可视化在地图上展示
        """)
    
    st.markdown("### 多项式回归练习")
    
    st.markdown("""
    尝试对以上线性回归模型进行改进，使用多项式回归来捕捉非线性特征关系。
    
    **任务**：
    1. 使用PolynomialFeatures转换特征
    2. 尝试不同的多项式阶数
    3. 对比多项式回归与线性回归的性能
    """)
    
    poly_code = st.text_area(
        "编辑多项式回归代码",
        """
# 导入所需库
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# 数据已加载在housing_df变量中
# X = housing_df.drop('PRICE', axis=1)
# y = housing_df['PRICE']

# 划分训练集和测试集
# X_train, X_test, y_train, y_test = ...

# 创建多项式回归Pipeline
# 包含标准化、多项式特征转换和线性回归
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('poly', PolynomialFeatures(degree=2, include_bias=False)),
#     ('linear', LinearRegression())
# ])

# 训练模型
# pipeline.fit(...)

# 预测和评估
# y_train_pred = ...
# y_test_pred = ...

# 计算性能指标
# 比较线性回归和多项式回归
        """,
        height=400
    )
    
    if st.button("运行多项式回归代码"):
        st.code(poly_code, language="python")
        
        # 示例解决方案
        st.markdown("### 多项式回归示例解决方案")
        
        poly_solution = """
# 导入所需库
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# 数据已加载在housing_df变量中
X = housing_df.drop('PRICE', axis=1)
y = housing_df['PRICE']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建多项式回归Pipeline
# 仅使用前3个最重要的特征，避免维度爆炸
important_features = ['MedInc', 'Latitude', 'Longitude']
X_train_important = X_train[important_features]
X_test_important = X_test[important_features]

# 不同阶数的多项式回归
degrees = [1, 2, 3]  # 1阶相当于线性回归
results = {}

for degree in degrees:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('linear', LinearRegression())
    ])
    
    # 训练模型
    pipeline.fit(X_train_important, y_train)
    
    # 预测
    y_train_pred = pipeline.predict(X_train_important)
    y_test_pred = pipeline.predict(X_test_important)
    
    # 计算性能指标
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    results[degree] = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse
    }
    
    print(f"{degree}阶多项式回归:")
    print(f"  训练集 R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
    print(f"  测试集 R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")
    print()
        """
        
        st.code(poly_solution, language="python")
        
        # 运行多项式回归示例并显示结果
        X = housing_df.drop('PRICE', axis=1)
        y = housing_df['PRICE']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        important_features = ['MedInc', 'Latitude', 'Longitude']
        X_train_important = X_train[important_features]
        X_test_important = X_test[important_features]
        
        degrees = [1, 2, 3]
        results = {}
        
        for degree in degrees:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                ('linear', LinearRegression())
            ])
            
            pipeline.fit(X_train_important, y_train)
            
            y_train_pred = pipeline.predict(X_train_important)
            y_test_pred = pipeline.predict(X_test_important)
            
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            results[degree] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse
            }
        
        # 显示结果对比
        st.markdown("#### 不同阶数多项式回归性能对比")
        
        results_df = pd.DataFrame({
            '多项式阶数': degrees,
            '训练集 R²': [results[d]['train_r2'] for d in degrees],
            '测试集 R²': [results[d]['test_r2'] for d in degrees],
            '训练集 RMSE': [results[d]['train_rmse'] for d in degrees],
            '测试集 RMSE': [results[d]['test_rmse'] for d in degrees]
        })
        
        st.dataframe(results_df)
        
        # 可视化结果对比
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1.plot(degrees, [results[d]['train_r2'] for d in degrees], 'o-', label='训练集')
        ax1.plot(degrees, [results[d]['test_r2'] for d in degrees], 'o-', label='测试集')
        ax1.set_xlabel('多项式阶数')
        ax1.set_ylabel('R²')
        ax1.set_title('R² vs 多项式阶数')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(degrees, [results[d]['train_rmse'] for d in degrees], 'o-', label='训练集')
        ax2.plot(degrees, [results[d]['test_rmse'] for d in degrees], 'o-', label='测试集')
        ax2.set_xlabel('多项式阶数')
        ax2.set_ylabel('RMSE')
        ax2.set_title('RMSE vs 多项式阶数')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        #### 多项式回归分析
        
        通过对比不同阶数的多项式回归模型性能，我们可以观察到：
        
        1. **性能提升**：
           - 从1阶（线性）到2阶多项式，模型性能有明显提升，R²增加，RMSE减小。
           - 从2阶到3阶，训练集性能继续提升，但测试集性能可能开始下降（过拟合的迹象）。
        
        2. **过拟合风险**：
           - 随着多项式阶数增加，训练集和测试集性能差距扩大，表明模型开始过拟合。
           - 高阶多项式（如3阶及以上）可能会捕捉数据中的噪声而非真实模式。
        
        3. **最佳选择**：
           - 在这个例子中，2阶多项式回归可能是最佳选择，它提供了比线性回归更好的拟合，同时避免了高阶多项式的过拟合风险。
           - 实际应用中，应该使用交叉验证来确定最佳的多项式阶数。
        """)
    
    # 提供更多实践资源
    st.markdown("### 进一步学习资源")
    
    st.markdown("""
    如果你想进一步深入学习回归算法，可以尝试以下资源和练习：
    
    1. **Scikit-learn官方教程**：[线性回归示例](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html)
    
    2. **Kaggle竞赛**：
       - [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
       - [Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand)
    
    3. **数据集**：
       - [波士顿房价数据集](https://www.kaggle.com/c/boston-housing)
       - [Ames房价数据集](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
       - [Wine Quality数据集](https://archive.ics.uci.edu/ml/datasets/wine+quality)
    
    4. **交互式学习平台**：
       - [DataCamp - 机器学习课程](https://www.datacamp.com/courses/supervised-learning-with-scikit-learn)
       - [Coursera - 吴恩达机器学习课程](https://www.coursera.org/learn/machine-learning)
    """)
    
if __name__ == "__main__":
    show() 