import streamlit as st
from exercises.ensemble_comparison import show_ensemble_comparison
from exercises.feature_importance import show_feature_importance_analysis
from exercises.model_tuning import show_model_tuning

def show():
    st.markdown('<h2 class="sub-header">高级练习</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    本节提供一些高级练习，帮助你深入理解决策树、随机森林和集成学习算法的高级主题。
    每个练习都提供交互式工具，让你能够实践和探索这些概念。
    """)
    
    # 创建选项卡
    tabs = st.tabs([
        "集成学习方法比较", 
        "特征重要性分析与特征选择", 
        "超参数调优与模型评估"
    ])
    
    # 集成学习方法比较选项卡
    with tabs[0]:
        show_ensemble_comparison()
    
    # 特征重要性分析与特征选择选项卡
    with tabs[1]:
        show_feature_importance_analysis()
    
    # 超参数调优与模型评估选项卡
    with tabs[2]:
        show_model_tuning() 