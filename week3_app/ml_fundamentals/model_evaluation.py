"""
模型评估与超参数选择的交互式演示模块
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from utils.svg_generator import create_dataset_split_svg, create_cross_validation_svg, create_roc_curve_svg, create_learning_curve_svg, render_svg
import os

def show_model_evaluation_demo():
    """显示模型评估与超参数选择的交互式演示"""
    
    st.subheader("模型评估与超参数选择")
    
    st.markdown("""
    正确评估机器学习模型的性能对于构建可靠、实用的模型至关重要。
    本节将介绍模型评估的基本方法、常用指标以及超参数调整的技术。
    """)
    
    # 创建选项卡
    option = st.radio(
        "选择主题:",
        ["数据分割策略", "交叉验证", "学习曲线", "评估指标", "超参数调整"],
        horizontal=True
    )
    
    if option == "数据分割策略":
        show_data_splitting()
    elif option == "交叉验证":
        show_cross_validation()
    elif option == "学习曲线":
        show_learning_curves()
    elif option == "评估指标":
        show_evaluation_metrics()
    elif option == "超参数调整":
        show_hyperparameter_tuning()

def show_data_splitting():
    """显示数据分割策略的内容"""
    
    st.markdown("### 数据分割策略")
    
    st.markdown("""
    在机器学习中，通常将数据集分为训练集、验证集和测试集：
    
    - **训练集**: 用于训练模型的数据
    - **验证集**: 用于调整模型参数和进行模型选择
    - **测试集**: 用于评估最终模型性能的独立数据集
    """)
    
    # 使用图片文件显示数据分割示意图
    split_img_path = os.path.join("img", "dataset_split.svg")
    
    # 检查图片是否存在，不存在则创建
    if not os.path.exists(split_img_path):
        # 确保img目录存在
        os.makedirs(os.path.dirname(split_img_path), exist_ok=True)
        
        # 创建SVG内容并保存到文件
        svg_content = create_dataset_split_svg()
        with open(split_img_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
    
    # 直接读取SVG内容并使用render_svg函数显示
    try:
        with open(split_img_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
        render_svg(svg_content)
    except Exception as e:
        st.error(f"显示SVG图片时出错: {str(e)}")
        st.warning("生成图片文件失败，请检查路径和权限")

def show_cross_validation():
    """显示交叉验证的内容"""
    
    st.markdown("### 交叉验证")
    
    st.markdown("""
    交叉验证是一种评估模型性能的方法，通过将数据分成多个子集，反复训练和评估模型，以获得更可靠的性能估计。
    
    最常用的是K折交叉验证(K-fold Cross Validation)：
    - 将数据分成K个大小相等的子集(折)
    - 每次使用K-1个子集进行训练，剩下的1个子集用于测试
    - 重复K次，每次使用不同的子集作为测试集
    - 最终结果是K次测试的平均值
    """)
    
    # 交互式控件
    k_folds = st.slider("选择折数 (K)", min_value=2, max_value=10, value=5, step=1)
    
    # 使用图片文件显示交叉验证示意图
    cv_img_path = os.path.join("img", f"cross_validation_k{k_folds}.svg")
    
    # 检查图片是否存在，不存在则创建
    if not os.path.exists(cv_img_path):
        # 确保img目录存在
        os.makedirs(os.path.dirname(cv_img_path), exist_ok=True)
        
        # 创建SVG内容并保存到文件
        svg_content = create_cross_validation_svg(k_folds)
        with open(cv_img_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
    
    # 直接读取SVG内容并使用render_svg函数显示
    try:
        with open(cv_img_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
        render_svg(svg_content)
    except Exception as e:
        st.error(f"显示SVG图片时出错: {str(e)}")
        st.warning("生成图片文件失败，请检查路径和权限")

def show_learning_curves():
    """显示学习曲线的内容"""
    
    st.markdown("### 学习曲线")
    
    st.markdown("""
    学习曲线显示了模型的训练误差和验证误差随训练数据量变化的趋势，帮助我们诊断模型是否存在偏差或方差问题。
    
    - **高偏差(欠拟合)**: 训练误差和验证误差都很高，且两者相近
    - **高方差(过拟合)**: 训练误差低但验证误差高，两者差距大
    - **理想情况**: 随着数据量增加，两者都收敛到较低的误差
    """)
    
    # 使用图片文件显示学习曲线示意图
    lc_img_path = os.path.join("img", "learning_curve.svg")
    
    # 检查图片是否存在，不存在则创建
    if not os.path.exists(lc_img_path):
        # 确保img目录存在
        os.makedirs(os.path.dirname(lc_img_path), exist_ok=True)
        
        # 创建SVG内容并保存到文件
        svg_content = create_learning_curve_svg()
        with open(lc_img_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
    
    # 直接读取SVG内容并使用render_svg函数显示
    try:
        with open(lc_img_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
        render_svg(svg_content)
    except Exception as e:
        st.error(f"显示SVG图片时出错: {str(e)}")
        st.warning("生成图片文件失败，请检查路径和权限")

def show_evaluation_metrics():
    """显示评估指标的内容"""
    
    st.markdown("### 模型评估指标")
    
    st.markdown("""
    选择合适的评估指标对于正确评估模型性能至关重要，不同的问题可能需要不同的评估指标。
    
    #### 分类问题常用指标:
    
    - **准确率(Accuracy)**: 正确预测的样本比例
    - **精确率(Precision)**: 预测为正类的样本中实际为正类的比例
    - **召回率(Recall)**: 实际为正类的样本中被正确预测为正类的比例
    - **F1分数**: 精确率和召回率的调和平均数
    - **ROC曲线**: 展示不同阈值下真正率与假正率的关系
    - **AUC**: ROC曲线下的面积，值越大越好
    
    #### 回归问题常用指标:
    
    - **均方误差(MSE)**: 预测值与真实值差的平方的平均值
    - **均方根误差(RMSE)**: MSE的平方根，与因变量单位相同
    - **平均绝对误差(MAE)**: 预测值与真实值差的绝对值的平均值
    - **决定系数(R²)**: 表示模型解释的因变量方差比例
    """)
    
    # 创建示例数据
    st.markdown("### ROC曲线示例")
    
    st.markdown("""
    ROC(接收者操作特征)曲线是评估二分类模型性能的重要工具，展示了随着分类阈值变化，真正率(TPR)与假正率(FPR)之间的关系。
    
    - **真正率/召回率(TPR/Recall)**: TP/(TP+FN)
    - **假正率(FPR)**: FP/(FP+TN)
    
    曲线下面积(AUC)值越接近1，模型性能越好。
    """)
    
    # 使用图片文件显示ROC曲线示意图
    roc_img_path = os.path.join("img", "roc_curve.svg")
    
    # 检查图片是否存在，不存在则创建
    if not os.path.exists(roc_img_path):
        # 确保img目录存在
        os.makedirs(os.path.dirname(roc_img_path), exist_ok=True)
        
        # 创建SVG内容并保存到文件
        svg_content = create_roc_curve_svg()
        with open(roc_img_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
    
    # 直接读取SVG内容并使用render_svg函数显示
    try:
        with open(roc_img_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
        render_svg(svg_content)
    except Exception as e:
        st.error(f"显示SVG图片时出错: {str(e)}")
        st.warning("生成图片文件失败，请检查路径和权限")

def show_hyperparameter_tuning():
    """显示超参数调整的内容"""
    
    st.markdown("### 超参数调整")
    
    st.markdown("""
    **超参数**是在模型训练开始前设置的参数，不同于模型在训练过程中学习的参数。对于分类算法，常见的超参数包括：
    
    - 逻辑回归：正则化强度C，正则化类型
    - SVM：正则化强度C，核函数类型，核函数参数(如gamma)
    
    合理的超参数选择和模型评估对于开发高性能机器学习模型至关重要。
    """)
    
    # 数据集划分
    st.markdown("### 数据集划分")
    
    st.markdown("""
    合理划分数据集对于超参数选择和模型评估至关重要：
    
    1. **训练集(Training Set)**：用于训练模型，模型直接学习这些数据
    
    2. **验证集(Validation Set)**：用于超参数调优和模型选择
    
    3. **测试集(Test Set)**：用于评估最终模型性能，只使用一次
    """)
    
    # 显示数据集划分图
    st.markdown(create_dataset_split_svg(), unsafe_allow_html=True)
    
    # 交叉验证
    st.markdown("### 交叉验证")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **交叉验证**是一种评估模型性能的技术，特别适用于数据量有限的情况。
        
        **k折交叉验证**的步骤：
        1. 将数据集随机分成k个相等大小的子集（折）
        2. 对每一折：
           - 使用该折作为验证集
           - 使用其余k-1折作为训练集
        3. 训练k个模型并计算平均性能
        
        **优点**：
        - 更稳健的性能估计
        - 减少对特定验证集的过拟合
        - 充分利用有限的数据
        """)
        
        # 添加k折数选择器
        k_folds = st.slider("选择k折数", min_value=3, max_value=10, value=5, step=1)
    
    with col2:
        # 显示交叉验证图
        st.markdown(create_cross_validation_svg(k_folds), unsafe_allow_html=True)
    
    # 学习曲线
    st.markdown("### 学习曲线")
    
    st.markdown("""
    **学习曲线**显示了模型性能如何随训练集大小变化。它是诊断偏差和方差问题的有力工具。
    
    - **高偏差(欠拟合)**: 训练误差和验证误差都很高，且两者接近
    - **高方差(过拟合)**: 训练误差低但验证误差高，两者差距大
    """)
    
    # 显示学习曲线图
    st.markdown(create_learning_curve_svg(), unsafe_allow_html=True)
    
    # 交互式演示：学习曲线
    st.markdown("### 交互式演示：学习曲线")
    
    st.markdown("""
    **学习曲线**显示模型性能如何随训练集大小变化。它帮助诊断偏差和方差问题：
    
    - **高偏差**：训练和验证性能都低，并且接近
    - **高方差**：训练性能高，但验证性能显著低于训练性能
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        c_param = st.slider(
            "正则化强度 (C)",
            min_value=0.001,
            max_value=10.0,
            value=1.0,
            step=0.1
        )
        
        penalty = st.radio(
            "正则化类型",
            ["l1", "l2"]
        )
        
        st.markdown("""
        **参数说明：**
        
        **C值**：正则化强度的倒数
        - 较小的C = 较强的正则化
        - 较大的C = 较弱的正则化
        
        **正则化类型：**
        - L1：倾向于产生稀疏解
        - L2：倾向于小权重值
        """)
    
    with col2:
        # 生成数据集
        X, y = make_classification(
            n_samples=300, n_features=20, n_informative=2, n_redundant=10,
            n_classes=2, random_state=42
        )
        
        # 计算学习曲线
        train_sizes = np.linspace(0.1, 1.0, 10)
        model = LogisticRegression(C=c_param, penalty=penalty, solver='liblinear', random_state=42)
        
        train_sizes, train_scores, valid_scores = learning_curve(
            model, X, y, train_sizes=train_sizes, cv=5, scoring='accuracy'
        )
        
        # 计算平均值和标准差
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        valid_mean = np.mean(valid_scores, axis=1)
        valid_std = np.std(valid_scores, axis=1)
        
        # 绘制学习曲线
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.grid(True, alpha=0.3)
        ax.fill_between(train_sizes, train_mean - train_std,
                       train_mean + train_std, alpha=0.1, color='blue')
        ax.fill_between(train_sizes, valid_mean - valid_std,
                       valid_mean + valid_std, alpha=0.1, color='red')
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='训练准确率')
        ax.plot(train_sizes, valid_mean, 'o-', color='red', label='交叉验证准确率')
        ax.set_xlabel('训练样本数量')
        ax.set_ylabel('准确率')
        ax.set_title(f'学习曲线 (LogisticRegression, C={c_param}, penalty={penalty})')
        ax.legend(loc='best')
        
        st.pyplot(fig)
        
        # 分析学习曲线
        gap = np.mean(train_mean - valid_mean)
        final_train = train_mean[-1]
        final_valid = valid_mean[-1]
        
        if final_train > 0.9 and gap > 0.1:
            diagnosis = "高方差(过拟合)：模型在训练集上表现很好，但验证集性能显著较低"
            suggestion = "尝试增加正则化强度，收集更多数据，或减少模型复杂度"
        elif final_train < 0.8 and gap < 0.1:
            diagnosis = "高偏差(欠拟合)：训练和验证性能都不理想，且接近"
            suggestion = "尝试减少正则化强度，使用更复杂的模型，或添加更多特征"
        else:
            diagnosis = "平衡：模型有合理的训练-验证性能权衡"
            suggestion = "当前参数设置似乎不错，可以微调以进一步改进"
        
        st.success(f"**诊断**: {diagnosis}")
        st.info(f"**建议**: {suggestion}")
    
    # 模型评估指标
    st.markdown("### 评估指标")
    
    st.markdown("""
    选择合适的评估指标对于理解模型性能至关重要：
    
    1. **准确率(Accuracy)**：正确预测的比例
       - 适用于平衡数据集
       - 公式：(TP + TN) / (TP + TN + FP + FN)
       
    2. **精确率(Precision)**：预测为正的样本中实际为正的比例
       - 衡量正类预测的准确性
       - 公式：TP / (TP + FP)
       
    3. **召回率(Recall)**：实际为正的样本中被正确预测的比例
       - 衡量模型发现正类的能力
       - 公式：TP / (TP + FN)
       
    4. **F1分数**：精确率和召回率的调和平均
       - 公式：2 * (Precision * Recall) / (Precision + Recall)
       
    5. **ROC曲线和AUC**：
       - 反映模型在不同决策阈值下的表现
       - AUC(曲线下面积)越接近1越好
    """)
    
    # 显示ROC曲线图
    st.markdown(create_roc_curve_svg(), unsafe_allow_html=True)
    
    # 超参数搜索方法
    st.markdown("### 超参数搜索方法")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("""
        **网格搜索(Grid Search)**
        
        系统地尝试所有参数组合。
        
        **步骤：**
        1. 定义参数值网格
        2. 对每个组合进行交叉验证
        3. 选择最佳组合
        
        **优点：**
        - 彻底、全面
        - 保证找到网格内的最佳点
        
        **缺点：**
        - 计算成本高
        - 受维度灾难影响
        """)
    
    with col2:
        st.markdown("""
        **随机搜索(Random Search)**
        
        从参数空间随机采样。
        
        **步骤：**
        1. 定义参数分布
        2. 随机采样n个组合
        3. 对每个组合进行交叉验证
        
        **优点：**
        - 比网格搜索更高效
        - 更好地覆盖高维空间
        
        **缺点：**
        - 不保证找到最优解
        - 可能需要多次运行
        """)
    
    with col3:
        st.markdown("""
        **贝叶斯优化**
        
        基于先前结果智能搜索。
        
        **步骤：**
        1. 构建性能的概率模型
        2. 根据期望改进选择下一组参数
        3. 更新模型并重复
        
        **优点：**
        - 高效利用计算资源
        - 适用于计算密集型任务
        
        **缺点：**
        - 实现复杂
        - 可能陷入局部最优
        """)
    
    # 验证曲线示例
    st.markdown("### 验证曲线示例")
    
    # 计算验证曲线
    param_range = np.logspace(-3, 3, 10)
    model = LogisticRegression(penalty='l2', solver='liblinear', random_state=42)
    
    train_scores, valid_scores = validation_curve(
        model, X, y, param_name="C", param_range=param_range,
        cv=5, scoring='accuracy'
    )
    
    # 计算平均值和标准差
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)
    
    # 绘制验证曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.grid(True, alpha=0.3)
    ax.semilogx(param_range, train_mean, 'o-', color='blue', label='训练准确率')
    ax.fill_between(param_range, train_mean - train_std,
                   train_mean + train_std, alpha=0.1, color='blue')
    ax.semilogx(param_range, valid_mean, 'o-', color='red', label='交叉验证准确率')
    ax.fill_between(param_range, valid_mean - valid_std,
                   valid_mean + valid_std, alpha=0.1, color='red')
    ax.set_xlabel('参数 C')
    ax.set_ylabel('准确率')
    ax.set_title('验证曲线 (LogisticRegression, l2惩罚)')
    ax.legend(loc='best')
    
    st.pyplot(fig)
    
    st.markdown("""
    **验证曲线分析：**
    
    验证曲线展示了模型性能如何随超参数变化。上图显示了逻辑回归模型的C参数如何影响训练和交叉验证准确率：
    
    - **较小的C值**（强正则化）可能导致欠拟合
    - **较大的C值**（弱正则化）可能导致过拟合
    - **最佳C值**在曲线中找到交叉验证准确率最高的点
    
    通过绘制各种超参数的验证曲线，可以直观地找到最佳参数设置，而不需要进行完整的网格搜索。
    """)
    
    # 实战练习提示
    st.info("""
    **实验任务**：
    1. 观察不同C值和正则化类型如何影响学习曲线
    2. 尝试找出给定数据集的最佳C值和正则化类型
    
    **思考问题**：
    1. 如何通过学习曲线判断模型是过拟合还是欠拟合？
    2. 如何在计算资源有限的情况下有效地选择超参数？
    """) 