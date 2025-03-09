import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
from sklearn.tree import export_graphviz
import pydot
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
from sklearn.metrics import confusion_matrix, roc_curve, auc
import platform

# 配置matplotlib支持中文显示，根据操作系统选择不同的字体
system = platform.system()
try:
    if system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
    elif system == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC', 'Hiragino Sans GB', 'STHeiti']
    elif system == 'Linux':
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK JP', 'Noto Sans CJK SC', 'Droid Sans Fallback']
    else:
        # 默认字体设置
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    matplotlib.rcParams['font.family'] = 'sans-serif'
except Exception as e:
    st.warning(f"无法设置中文字体: {e}。图表中的中文可能无法正确显示。")

def plot_decision_tree(model, feature_names, class_names, max_depth=None):
    """可视化决策树模型"""
    dot_data = export_graphviz(
        model, 
        out_file=None,
        max_depth=max_depth,
        feature_names=feature_names,
        class_names=class_names,
        filled=True, 
        rounded=True,
        special_characters=True
    )
    
    graph = graphviz.Source(dot_data)
    st.graphviz_chart(dot_data)
    
    return graph

def plot_feature_importance(model, feature_names, top_n=10):
    """可视化特征重要性"""
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    if top_n and len(feature_imp) > top_n:
        feature_imp = feature_imp.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_imp, ax=ax)
    ax.set_title('特征重要性')
    ax.set_xlabel('重要性')
    ax.set_ylabel('特征')
    
    st.pyplot(fig)
    
    return feature_imp

def plot_confusion_matrix(cm, class_names=None):
    """可视化混淆矩阵"""
    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]
    
    fig = px.imshow(
        cm, 
        x=class_names, 
        y=class_names,
        color_continuous_scale='Blues',
        labels=dict(x="预测标签", y="真实标签", color="数量"),
        text_auto=True
    )
    fig.update_layout(title="混淆矩阵")
    
    st.plotly_chart(fig)

def plot_roc_curve(fpr, tpr, auc_score):
    """可视化ROC曲线"""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr, 
            y=tpr,
            mode='lines',
            name=f'ROC曲线 (AUC = {auc_score:.3f})'
        )
    )
    
    # 添加对角线（随机猜测的基准线）
    fig.add_trace(
        go.Scatter(
            x=[0, 1], 
            y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='随机猜测'
        )
    )
    
    fig.update_layout(
        title='ROC曲线',
        xaxis_title='假正例率 (FPR)',
        yaxis_title='真正例率 (TPR)',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    
    st.plotly_chart(fig)

def plot_learning_curves(train_scores, test_scores, train_sizes):
    """可视化学习曲线"""
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    fig = go.Figure()
    
    # 添加训练集得分
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean,
        mode='lines+markers',
        name='训练集得分',
        line=dict(color='blue'),
    ))
    
    # 添加训练集得分的上下界
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.1)',
        line=dict(color='rgba(0, 0, 255, 0)'),
        name='训练集标准差'
    ))
    
    # 添加测试集得分
    fig.add_trace(go.Scatter(
        x=train_sizes, y=test_mean,
        mode='lines+markers',
        name='测试集得分',
        line=dict(color='red'),
    ))
    
    # 添加测试集得分的上下界
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([test_mean + test_std, (test_mean - test_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.1)',
        line=dict(color='rgba(255, 0, 0, 0)'),
        name='测试集标准差'
    ))
    
    fig.update_layout(
        title='学习曲线',
        xaxis_title='训练样本数',
        yaxis_title='得分',
        xaxis=dict(tickformat=','),
        legend=dict(x=0.01, y=0.01, bgcolor='rgba(255, 255, 255, 0.5)'),
    )
    
    st.plotly_chart(fig)

def plot_grid_search_results(grid_results, param_name):
    """可视化网格搜索结果"""
    mean_scores = grid_results['mean_test_score']
    std_scores = grid_results['std_test_score']
    
    # 修复处理参数值的方式，避免KeyError
    # 旧代码：
    # params = grid_results['params']
    # param_values = [p[param_name] for p in params]
    
    # 新的改进代码：直接使用网格搜索结果中的参数值
    param_values = grid_results[param_name]
    
    # 由于param_values可能包含None值，我们需要将其转换为可排序的形式
    # 对于数值类型，保持不变；对于None值或其他类型，转换为字符串
    param_values_display = [str(val) if val is None or not isinstance(val, (int, float)) else val 
                           for val in param_values]
    
    # 为了正确显示，我们需要同时对分数和参数值进行排序
    sorted_indices = np.argsort(param_values_display)
    sorted_param_values = [param_values_display[i] for i in sorted_indices]
    sorted_mean_scores = [mean_scores[i] for i in sorted_indices]
    sorted_std_scores = [std_scores[i] for i in sorted_indices]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=sorted_param_values,
        y=sorted_mean_scores,
        error_y=dict(
            type='data',
            array=sorted_std_scores,
            visible=True
        ),
        mode='lines+markers',
        name='验证得分'
    ))
    
    fig.update_layout(
        title=f'参数 {param_name} 的网格搜索结果',
        xaxis_title=param_name,
        yaxis_title='验证得分',
    )
    
    st.plotly_chart(fig)
    
def plot_cross_validation(cv_scores):
    """可视化交叉验证结果"""
    fold_indices = np.arange(1, len(cv_scores) + 1)
    mean_score = np.mean(cv_scores)
    
    fig = go.Figure()
    
    # 添加每个折叠的得分
    fig.add_trace(go.Bar(
        x=fold_indices,
        y=cv_scores,
        name='折叠得分',
        marker_color='royalblue'
    ))
    
    # 添加平均得分线
    fig.add_trace(go.Scatter(
        x=[0.5, len(cv_scores) + 0.5],
        y=[mean_score, mean_score],
        mode='lines',
        name=f'平均得分: {mean_score:.3f}',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='交叉验证结果',
        xaxis_title='折叠',
        yaxis_title='得分',
        xaxis=dict(tickmode='linear', dtick=1),
    )
    
    st.plotly_chart(fig) 