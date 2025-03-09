import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_blobs

def create_styled_container(title, content, style="info"):
    """创建一个样式化的容器"""
    if style == "info":
        st.markdown(f'<div class="info-box"><h3>{title}</h3>{content}</div>', unsafe_allow_html=True)
    elif style == "objective":
        st.markdown(f'<div class="objective-box"><h3>{title}</h3>{content}</div>', unsafe_allow_html=True)
    elif style == "warning":
        st.markdown(f'<div class="warning-box"><h3>{title}</h3>{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<h3>{title}</h3>{content}', unsafe_allow_html=True)

def generate_sample_data(n_samples=1000, n_features=10, n_classes=2, random_state=42):
    """生成示例分类数据集"""
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=n_features, 
        n_informative=5, 
        n_redundant=2, 
        n_classes=n_classes, 
        random_state=random_state
    )
    
    # 创建DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df

def generate_ecommerce_data(n_samples=1000, random_state=42):
    """生成模拟电商用户行为数据"""
    np.random.seed(random_state)
    
    # 用户基本信息
    user_id = np.arange(1, n_samples + 1)
    age = np.random.normal(35, 10, n_samples).astype(int)
    age = np.clip(age, 18, 70)  # 限制年龄范围
    
    # 0=女性, 1=男性
    gender = np.random.binomial(1, 0.5, n_samples)
    
    # 用户行为数据
    visit_frequency = np.random.poisson(10, n_samples)  # 访问频率
    avg_time_spent = np.random.gamma(5, 2, n_samples)  # 平均停留时间(分钟)
    items_viewed = np.random.poisson(15, n_samples)  # 浏览商品数
    items_purchased = np.random.binomial(items_viewed, 0.3, n_samples)  # 购买商品数
    
    # 购物车相关
    cart_abandonment = np.random.binomial(1, 0.4, n_samples)  # 购物车放弃率
    cart_value = np.random.gamma(100, 10, n_samples)  # 购物车价值
    
    # 用户历史
    purchase_history = np.random.gamma(500, 20, n_samples)  # 历史购买金额
    return_rate = np.random.beta(2, 10, n_samples)  # 退货率
    
    # 生成目标变量: 高价值(1)vs低价值(0)用户
    # 基于用户行为指标计算一个用户价值分数
    value_score = (
        visit_frequency * 0.2 + 
        items_purchased * 0.3 + 
        purchase_history * 0.002 - 
        cart_abandonment * 5 - 
        return_rate * 10
    )
    
    # 将分数转换为二元目标
    threshold = np.percentile(value_score, 70)  # 设置为前30%为高价值用户
    user_value = (value_score > threshold).astype(int)
    
    # 创建DataFrame
    data = {
        'user_id': user_id,
        'age': age,
        'gender': gender,
        'visit_frequency': visit_frequency,
        'avg_time_spent': avg_time_spent,
        'items_viewed': items_viewed,
        'items_purchased': items_purchased,
        'cart_abandonment': cart_abandonment,
        'cart_value': cart_value,
        'purchase_history': purchase_history,
        'return_rate': return_rate,
        'high_value_user': user_value
    }
    
    df = pd.DataFrame(data)
    return df 