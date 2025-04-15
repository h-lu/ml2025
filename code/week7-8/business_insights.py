import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 设置matplotlib支持中文显示
# 根据操作系统设置合适的中文字体
system = platform.system()
if system == 'Darwin':  # macOS
    plt.rcParams['font.family'] = ['Arial Unicode MS', 'PingFang SC', 'STHeiti', 'Heiti TC', 'sans-serif']
elif system == 'Windows':
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'sans-serif']
else:  # Linux或其他系统
    plt.rcParams['font.family'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'sans-serif']

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

def generate_user_data(n_samples=500, random_state=42):
    """生成模拟用户行为数据"""
    rng = np.random.RandomState(random_state)
    
    # 创建用户特征
    data = {
        # 人口统计学特征
        'age': rng.normal(35, 12, n_samples).clip(18, 80).astype(int),
        'gender': rng.choice(['Male', 'Female'], n_samples),
        'income': rng.normal(50000, 20000, n_samples).clip(10000, 150000).astype(int),
        
        # 消费行为特征
        'purchase_frequency': rng.normal(5, 3, n_samples).clip(0, 20).astype(int),  # 月均购买次数
        'average_order_value': rng.normal(100, 60, n_samples).clip(10, 500).astype(int),  # 平均订单金额
        'total_spend': rng.normal(2000, 1500, n_samples).clip(0, 10000).astype(int),  # 总消费金额
        
        # 用户活跃度特征
        'days_since_last_purchase': rng.normal(30, 20, n_samples).clip(0, 365).astype(int),  # 最近购买间隔
        'browsing_time_mins': rng.normal(120, 60, n_samples).clip(0, 300).astype(int),  # 月均浏览时长(分钟)
        'login_count': rng.normal(15, 10, n_samples).clip(0, 60).astype(int),  # 月均登录次数
        
        # 产品偏好特征
        'product_category_preference': rng.choice(['Electronics', 'Clothing', 'Home', 'Beauty', 'Books'], n_samples),
        'discount_sensitivity': rng.normal(5, 2, n_samples).clip(1, 10).astype(int),  # 1-10分，越高越敏感
    }
    
    # 创建一些有相关性的特征
    # 高收入用户平均订单金额往往更高
    data['average_order_value'] = (data['average_order_value'] + data['income'] / 1000 * rng.normal(0.5, 0.1, n_samples)).astype(int)
    
    # 活跃用户(登录多)往往浏览时间更长，购买频率更高
    data['browsing_time_mins'] = (data['browsing_time_mins'] + data['login_count'] * 2 * rng.normal(1, 0.1, n_samples)).astype(int)
    data['purchase_frequency'] = (data['purchase_frequency'] + data['login_count'] / 5 * rng.normal(1, 0.2, n_samples)).astype(int)
    
    # 总消费 = 购买频率 * 平均订单金额 (加一些随机波动)
    data['total_spend'] = (data['purchase_frequency'] * data['average_order_value'] * rng.normal(1, 0.2, n_samples)).astype(int)
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    return df

def business_insights_demo():
    """业务洞察解读演示页面"""
    st.header("聚类结果的业务解读")
    
    st.write("""
    在实际业务场景中，聚类算法的最终目的不仅是将数据分组，更重要的是从这些分组中提取有价值的业务洞察，
    并将其转化为具体的商业策略和行动。这个环节至关重要，是将数据分析转化为业务价值的关键步骤。
    """)
    
    # 顾客分群案例
    st.subheader("用户分群案例")
    
    # 生成数据
    if "user_data" not in st.session_state:
        n_samples = 500
        st.session_state.user_data = generate_user_data(n_samples)
    
    user_df = st.session_state.user_data
    
    # 显示原始数据
    with st.expander("查看原始用户数据"):
        st.dataframe(user_df.head(10))
        
        # 显示基本统计信息
        st.write("基本统计信息:")
        st.dataframe(user_df.describe().T)
    
    # 可视化一些关键特征的分布
    with st.expander("查看数据特征分布"):
        feature_to_plot = st.selectbox(
            "选择特征:",
            ["age", "income", "purchase_frequency", "average_order_value", 
             "total_spend", "browsing_time_mins", "days_since_last_purchase"]
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=user_df, x=feature_to_plot, kde=True, ax=ax)
        ax.set_title(f"{feature_to_plot}分布")
        st.pyplot(fig)
        
        # 散点图
        col1, col2 = st.columns(2)
        
        with col1:
            x_feature = st.selectbox("X轴特征:", 
                                   ["age", "income", "purchase_frequency", "average_order_value"], key="x_feature")
        
        with col2:
            y_feature = st.selectbox("Y轴特征:", 
                                   ["total_spend", "browsing_time_mins", "login_count", "days_since_last_purchase"], key="y_feature")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=user_df, x=x_feature, y=y_feature, hue="gender", alpha=0.7, ax=ax)
        ax.set_title(f"{x_feature} vs {y_feature}")
        st.pyplot(fig)
    
    # 聚类分析部分
    st.subheader("用户聚类分析")
    
    # 选择聚类特征
    feature_options = [
        "age", "income", "purchase_frequency", "average_order_value", 
        "total_spend", "browsing_time_mins", "login_count", 
        "days_since_last_purchase", "discount_sensitivity"
    ]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        selected_features = st.multiselect(
            "选择用于聚类的特征:",
            feature_options,
            default=["age", "income", "purchase_frequency", "average_order_value", "browsing_time_mins"]
        )
        
        if not selected_features:
            st.warning("请至少选择一个特征进行聚类")
            return
    
    with col2:
        cluster_method = st.radio("选择聚类算法:", ["K-Means", "DBSCAN"])
        
        if cluster_method == "K-Means":
            n_clusters = st.slider("簇数量K:", 2, 10, 4)
        else:
            eps = st.slider("eps值:", 0.1, 2.0, 0.5, 0.1)
            min_samples = st.slider("min_samples值:", 2, 20, 5)
    
    # 处理选择的特征
    X = user_df[selected_features].copy()
    
    # 特征预处理 (缩放)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 执行聚类
    if cluster_method == "K-Means":
        clustering = KMeans(n_clusters=n_clusters, random_state=42)
        user_df['cluster'] = clustering.fit_predict(X_scaled)
        cluster_col = 'cluster'
        
        centers = scaler.inverse_transform(clustering.cluster_centers_)
        centers_df = pd.DataFrame(centers, columns=selected_features)
        centers_df.index.name = 'cluster'
        centers_df.reset_index(inplace=True)
        
        # 绘制每个簇的特征平均值雷达图
        if len(selected_features) >= 3:
            st.subheader("各簇特征分布")
            
            # 准备绘制雷达图的数据
            # 首先，归一化特征便于雷达图展示
            min_max_scaler = MinMaxScaler()
            centers_radar = centers_df.copy()
            centers_radar[selected_features] = min_max_scaler.fit_transform(centers_radar[selected_features])
            
            # 绘制雷达图
            fig = plt.figure(figsize=(12, 8))
            
            # 设置雷达图参数
            N = len(selected_features)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # 闭合雷达图
            
            # 创建子图位置
            if n_clusters <= 4:
                rows, cols = 2, 2
            else:
                rows, cols = 3, 3
            
            # 绘制每个簇的雷达图
            for i in range(n_clusters):
                ax = plt.subplot(rows, cols, i+1, polar=True)
                
                # 添加每个特征的刻度和标签
                plt.xticks(angles[:-1], selected_features, size=8)
                
                # 绘制雷达图轮廓
                values = centers_radar[centers_radar['cluster'] == i][selected_features].values.flatten().tolist()
                values += values[:1]  # 闭合雷达图
                ax.plot(angles, values, linewidth=1, linestyle='solid')
                ax.fill(angles, values, alpha=0.1)
                
                # 添加簇标签
                plt.title(f'簇 {i}', size=11)
            
            plt.tight_layout()
            st.pyplot(fig)
            
    else:  # DBSCAN
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        user_df['cluster'] = clustering.fit_predict(X_scaled)
        cluster_col = 'cluster'
        
        # 计算簇数量和噪声点数量
        n_clusters = len(set(user_df['cluster'])) - (1 if -1 in user_df['cluster'] else 0)
        n_noise = list(user_df['cluster']).count(-1)
        st.write(f"DBSCAN发现的簇数量: {n_clusters}")
        st.write(f"噪声点数量: {n_noise} ({n_noise/len(user_df):.1%})")
    
    # 展示各簇主要特征
    st.subheader("各簇特征统计")
    
    # 每个簇的大小
    cluster_sizes = user_df[cluster_col].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    cluster_sizes.plot(kind='bar', ax=ax)
    ax.set_title("各簇大小")
    ax.set_xlabel("簇")
    ax.set_ylabel("用户数量")
    st.pyplot(fig)
    
    # 计算每个簇的特征均值
    cluster_means = user_df.groupby(cluster_col)[selected_features].mean().round(2)
    
    # 添加簇大小信息
    cluster_means['用户数量'] = cluster_sizes.values
    cluster_means['用户占比'] = (cluster_sizes / cluster_sizes.sum() * 100).round(1).astype(str) + '%'
    
    # 显示聚类结果
    st.dataframe(cluster_means)
    
    # 特征箱线图
    st.subheader("各簇特征比较")
    
    feature_for_boxplot = st.selectbox(
        "选择要比较的特征:",
        selected_features
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=user_df, x=cluster_col, y=feature_for_boxplot, ax=ax)
    ax.set_title(f"各簇 {feature_for_boxplot} 分布")
    ax.set_xlabel("簇")
    st.pyplot(fig)
    
    # 分类特征分布
    categorial_feature = st.selectbox(
        "选择要分析的分类特征:",
        ["gender", "product_category_preference"]
    )
    
    # 计算每个簇中的分类特征分布
    cat_dist = pd.crosstab(
        user_df[cluster_col], 
        user_df[categorial_feature], 
        normalize='index'
    ) * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    cat_dist.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title(f"各簇 {categorial_feature} 分布")
    ax.set_xlabel("簇")
    ax.set_ylabel("百分比 (%)")
    ax.legend(title=categorial_feature)
    st.pyplot(fig)
    
    # 业务洞察与解读部分
    st.subheader("用户群体画像与营销策略")
    
    st.write("""
    通过聚类分析，我们可以为不同用户群体创建特征画像，并制定针对性的营销策略。
    """)
    
    # 根据聚类结果给簇贴标签
    if cluster_method == "K-Means" and n_clusters == 4:
        st.markdown("""
        ### 用户群体标签及解读
        
        以下是对每个群体的解读和建议的营销策略，用于演示目的：
        
        | 群体 | 特征描述 | 营销策略 |
        | --- | --- | --- |
        | **高价值核心用户** | 中高年龄、高收入、高消费频率与金额 | 提供VIP服务与专属优惠，强调高品质产品，维护长期忠诚度 |
        | **潜力年轻人** | 年轻、中等收入、中高消费意愿 | 推送新潮产品、限时折扣，利用社交媒体营销 |
        | **保守型用户** | 较高年龄、中等收入、低消费频率 | 推荐实用性强的产品，强调性价比和耐用性 |
        | **低活跃/流失风险用户** | 各年龄段、低登录次数、低消费 | 开展召回活动，提供回归奖励，分析流失原因 |
        """)
    else:
        st.write("""
        #### 创建用户群体标签
        
        根据聚类结果，您可以给每个簇起一个有商业意义的名称，概括群体特征：
        
        1. 先分析每个簇的显著特征（高于或低于平均值的指标）
        2. 基于这些特征，起一个简洁、直观的名称
        3. 确保名称反映该群体的核心商业价值
        
        #### 制定针对性营销策略
        
        针对不同群体，可以制定差异化的营销策略：
        
        - **高价值用户群体:** 强调品质和专属性，减少价格敏感内容
        - **价格敏感群体:** 提供折扣和优惠券，强调性价比
        - **流失风险用户:** 设计召回活动，提供特别优惠
        - **新用户/潜力用户:** 精简入门流程，提供新手教程和首单优惠
        """)
    
    # 决策建议与结论
    with st.expander("聚类分析应用场景"):
        st.write("""
        ### 用户分群分析的商业应用
        
        - **个性化营销:** 为不同用户群体定制不同的营销信息和促销活动
        - **产品开发:** 基于不同群体需求开发或调整产品功能
        - **客户服务:** 为高价值用户群体提供更高级别的服务
        - **定价策略:** 根据不同群体的价格敏感度调整定价和折扣策略
        - **客户留存:** 识别流失风险客户，制定挽留计划
        - **用户体验优化:** 根据不同群体的行为模式优化用户界面和流程
        - **库存管理:** 基于不同群体的购买偏好调整库存策略
        
        ### 关键成功因素
        
        1. **选择合适的特征:** 确保所选特征能够反映用户的真实行为和价值
        2. **适当的聚类算法和参数:** 根据数据特点选择合适的算法
        3. **业务理解:** 将技术结果转化为有意义的业务洞察
        4. **行动导向:** 确保分析结果能转化为具体行动
        5. **持续监测:** 定期更新分群分析，反映用户行为变化
        """)
    
    st.write("""
    > **注意:** 用户分群分析应该是一个持续优化的过程，而不是一次性项目。随着业务变化和用户行为演变，聚类结果可能需要定期更新。
    """) 