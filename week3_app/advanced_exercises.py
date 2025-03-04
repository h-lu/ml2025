import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import time
import os

def load_data():
    """加载数据集，这里默认使用已经下载好的数据"""
    data_path = os.path.join("data", "adult.csv")
    
    try:
        df = pd.read_csv(data_path)
        # 如果数据加载成功，返回数据框
        return df
    except:
        # 如果数据不存在，显示错误消息
        st.error(f"数据文件 {data_path} 不存在！请先下载数据。")
        return None

def show_income_prediction():
    """显示收入预测任务"""
    st.subheader("任务1: 收入预测")
    
    st.markdown("""
    ### 背景介绍
    
    在这个任务中，我们将使用UCI成人人口普查收入数据集（Adult Census Income）来预测一个人的收入是否超过50K/年。
    这是一个真实世界的二分类问题，数据包含人口统计和就业相关特征。
    
    ### 数据集信息
    
    - **目标变量**: 收入是否 >50K/年（二元分类）
    - **特征**: 年龄、工作类型、教育程度、婚姻状况、职业、种族、性别等
    - **样本数**: 约32,561条记录
    - **挑战**: 处理混合的数值和分类特征、缺失值和不平衡类别
    """)
    
    # 检查数据是否存在
    data_exists = os.path.exists(os.path.join("data", "adult.csv"))
    
    if not data_exists:
        st.warning("数据文件不存在。请先下载数据集。")
        if st.button("下载数据集"):
            # 创建data目录
            os.makedirs("data", exist_ok=True)
            
            # 使用pandas直接从URL下载数据
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            column_names = [
                "age", "workclass", "fnlwgt", "education", "education-num", 
                "marital-status", "occupation", "relationship", "race", "sex", 
                "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
            ]
            
            try:
                df = pd.read_csv(url, names=column_names, sep=", ", engine='python')
                df.to_csv(os.path.join("data", "adult.csv"), index=False)
                st.success("数据下载成功！")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"下载数据时出错: {e}")
    else:
        # 加载数据
        df = load_data()
        
        if df is not None:
            # 显示数据集预览
            st.markdown("### 数据预览")
            st.dataframe(df.head())
            
            # 数据集统计信息
            st.markdown("### 数据集统计信息")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**样本数**: {df.shape[0]}")
                st.markdown(f"**特征数**: {df.shape[1] - 1}")  # 减去目标变量
                
                # 类别分布
                income_counts = df['income'].value_counts()
                st.markdown("**类别分布**")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.countplot(x='income', data=df, ax=ax)
                st.pyplot(fig)
            
            with col2:
                # 显示数据类型信息
                st.markdown("**数据类型**")
                st.dataframe(pd.DataFrame(df.dtypes, columns=["数据类型"]))
            
            # 数据预处理和特征工程
            st.markdown("### 数据预处理和特征工程")
            
            # 选择要处理的特征
            numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
            categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
            
            with st.expander("特征选择"):
                selected_numerical = st.multiselect(
                    "选择数值特征",
                    numerical_cols,
                    default=numerical_cols
                )
                
                selected_categorical = st.multiselect(
                    "选择分类特征",
                    categorical_cols,
                    default=categorical_cols[:4]  # 默认选择前4个分类特征
                )
            
            # 处理缺失值
            with st.expander("缺失值处理"):
                # 替换'?'为NaN
                df_clean = df.copy()
                for col in categorical_cols:
                    df_clean[col] = df_clean[col].replace(' ?', np.nan)
                
                # 显示缺失值统计
                missing_data = df_clean.isnull().sum()
                missing_data = missing_data[missing_data > 0]
                
                st.markdown("**缺失值统计**")
                if len(missing_data) > 0:
                    st.dataframe(pd.DataFrame(missing_data, columns=["缺失值数量"]))
                    
                    # 缺失值处理方法
                    missing_method = st.radio(
                        "选择缺失值处理方法",
                        ["删除含有缺失值的行", "用众数填充缺失值"]
                    )
                    
                    if missing_method == "删除含有缺失值的行":
                        df_clean = df_clean.dropna()
                        st.markdown(f"删除后的样本数: {df_clean.shape[0]}")
                    else:
                        for col in missing_data.index:
                            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                        st.markdown("已用众数填充缺失值")
                else:
                    st.markdown("数据集中没有缺失值")
            
            # 特征编码
            with st.expander("特征编码"):
                encoding_method = st.radio(
                    "选择分类特征编码方法",
                    ["One-Hot编码", "Label编码"]
                )
            
            # 模型参数设置
            st.markdown("### 模型参数设置")
            
            model_type = st.radio(
                "选择模型",
                ["逻辑回归", "SVM", "模型比较"]
            )
            
            use_grid_search = st.checkbox("使用网格搜索进行超参数调优", value=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                test_size = st.slider("测试集比例", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
                random_state = st.slider("随机种子", min_value=0, max_value=100, value=42, step=1)
            
            with col2:
                # 不同模型的参数
                if model_type in ["逻辑回归", "模型比较"]:
                    if use_grid_search:
                        st.markdown("**逻辑回归网格搜索参数**")
                        lr_C_min = st.slider("C最小值(Log10)", min_value=-3, max_value=0, value=-2, step=1, key="lr_c_min")
                        lr_C_max = st.slider("C最大值(Log10)", min_value=0, max_value=3, value=2, step=1, key="lr_c_max")
                    else:
                        st.markdown("**逻辑回归参数**")
                        lr_C = st.slider("正则化强度C", min_value=0.01, max_value=10.0, value=1.0, step=0.1, key="lr_C")
                        lr_solver = st.selectbox("优化算法", ["liblinear", "lbfgs", "newton-cg", "saga"], key="lr_solver")
                
                if model_type in ["SVM", "模型比较"]:
                    if use_grid_search:
                        st.markdown("**SVM网格搜索参数**")
                        svm_kernel = st.selectbox("核函数", ["linear", "rbf"], key="svm_k_gs")
                        svm_C_min = st.slider("C最小值(Log10)", min_value=-3, max_value=0, value=-2, step=1, key="svm_c_min")
                        svm_C_max = st.slider("C最大值(Log10)", min_value=0, max_value=3, value=2, step=1, key="svm_c_max")
                        
                        if svm_kernel == "rbf":
                            svm_gamma_min = st.slider("Gamma最小值(Log10)", min_value=-3, max_value=0, value=-2, step=1)
                            svm_gamma_max = st.slider("Gamma最大值(Log10)", min_value=0, max_value=3, value=1, step=1)
                    else:
                        st.markdown("**SVM参数**")
                        svm_C = st.slider("正则化强度C", min_value=0.01, max_value=10.0, value=1.0, step=0.1, key="svm_C")
                        svm_kernel = st.selectbox("核函数", ["linear", "rbf"], key="svm_kernel")
                        
                        if svm_kernel == "rbf":
                            svm_gamma = st.slider("Gamma参数", min_value=0.01, max_value=10.0, value=0.1, step=0.01)
            
            # 训练模型
            if st.button("训练模型"):
                # 准备数据
                # 处理目标变量
                # 检查目标变量的分布
                income_distribution = df_clean['income'].value_counts()
                st.write("收入分布:", income_distribution)
                
                # 确保'>50K'和'<=50K'的格式一致
                y = df_clean['income'].apply(lambda x: 1 if '>50K' in x else 0)
                
                # 显示标签分布
                st.write("处理后的标签分布:", y.value_counts())
                
                # 提取和处理特征
                X_num = df_clean[selected_numerical].copy()
                
                # 分类特征编码
                if encoding_method == "One-Hot编码":
                    X_cat = pd.get_dummies(df_clean[selected_categorical], drop_first=True)
                else:  # Label编码
                    X_cat = df_clean[selected_categorical].copy()
                    for col in selected_categorical:
                        le = LabelEncoder()
                        X_cat[col] = le.fit_transform(df_clean[col].astype(str))
                
                # 合并特征
                if len(selected_categorical) > 0:
                    X = pd.concat([X_num, X_cat], axis=1)
                else:
                    X = X_num
                
                # 分割数据
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
                
                # 数据标准化
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # 创建结果容器
                results = {}
                
                # 使用进度条
                progress_bar = st.progress(0)
                
                # 训练和评估模型
                if model_type in ["逻辑回归", "模型比较"]:
                    with st.spinner("训练逻辑回归模型..."):
                        start_time = time.time()
                        
                        if use_grid_search:
                            # 设置网格搜索参数
                            lr_param_grid = {
                                'C': np.logspace(lr_C_min, lr_C_max, 10)
                            }
                            
                            # 创建模型和网格搜索
                            lr_model = LogisticRegression(random_state=random_state, max_iter=1000)
                            lr_grid = GridSearchCV(lr_model, lr_param_grid, cv=5, scoring='accuracy')
                            
                            # 训练模型
                            lr_grid.fit(X_train_scaled, y_train)
                            
                            # 获取最佳模型和参数
                            lr_best_model = lr_grid.best_estimator_
                            lr_best_params = lr_grid.best_params_
                            
                            # 模型预测
                            y_pred_lr = lr_best_model.predict(X_test_scaled)
                            y_pred_proba_lr = lr_best_model.predict_proba(X_test_scaled)[:, 1]
                            
                            # 计算性能指标
                            lr_accuracy = accuracy_score(y_test, y_pred_lr)
                            lr_cm = confusion_matrix(y_test, y_pred_lr)
                            lr_report = classification_report(y_test, y_pred_lr)
                            
                            # 保存结果
                            results['逻辑回归'] = {
                                'model': lr_best_model,
                                'params': lr_best_params,
                                'accuracy': lr_accuracy,
                                'confusion_matrix': lr_cm,
                                'report': lr_report,
                                'y_pred': y_pred_lr,
                                'y_pred_proba': y_pred_proba_lr,
                                'training_time': time.time() - start_time
                            }
                        else:
                            # 创建和训练模型
                            lr_model = LogisticRegression(C=lr_C, solver=lr_solver, random_state=random_state, max_iter=1000)
                            lr_model.fit(X_train_scaled, y_train)
                            
                            # 模型预测
                            y_pred_lr = lr_model.predict(X_test_scaled)
                            y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
                            
                            # 计算性能指标
                            lr_accuracy = accuracy_score(y_test, y_pred_lr)
                            lr_cm = confusion_matrix(y_test, y_pred_lr)
                            lr_report = classification_report(y_test, y_pred_lr)
                            
                            # 保存结果
                            results['逻辑回归'] = {
                                'model': lr_model,
                                'accuracy': lr_accuracy,
                                'confusion_matrix': lr_cm,
                                'report': lr_report,
                                'y_pred': y_pred_lr,
                                'y_pred_proba': y_pred_proba_lr,
                                'training_time': time.time() - start_time
                            }
                            
                    progress_bar.progress(50 if model_type == "模型比较" else 100)
                
                if model_type in ["SVM", "模型比较"]:
                    with st.spinner("训练SVM模型..."):
                        start_time = time.time()
                        
                        if use_grid_search:
                            # 设置网格搜索参数
                            if svm_kernel == "rbf":
                                svm_param_grid = {
                                    'C': np.logspace(svm_C_min, svm_C_max, 5),
                                    'gamma': np.logspace(svm_gamma_min, svm_gamma_max, 5)
                                }
                            else:
                                svm_param_grid = {
                                    'C': np.logspace(svm_C_min, svm_C_max, 10)
                                }
                            
                            # 创建模型和网格搜索
                            svm_model = SVC(kernel=svm_kernel, probability=True, random_state=random_state)
                            svm_grid = GridSearchCV(svm_model, svm_param_grid, cv=3, scoring='accuracy')
                            
                            # 训练模型
                            svm_grid.fit(X_train_scaled, y_train)
                            
                            # 获取最佳模型和参数
                            svm_best_model = svm_grid.best_estimator_
                            svm_best_params = svm_grid.best_params_
                            
                            # 模型预测
                            y_pred_svm = svm_best_model.predict(X_test_scaled)
                            y_pred_proba_svm = svm_best_model.predict_proba(X_test_scaled)[:, 1]
                            
                            # 计算性能指标
                            svm_accuracy = accuracy_score(y_test, y_pred_svm)
                            svm_cm = confusion_matrix(y_test, y_pred_svm)
                            svm_report = classification_report(y_test, y_pred_svm)
                            
                            # 保存结果
                            results['SVM'] = {
                                'model': svm_best_model,
                                'params': svm_best_params,
                                'accuracy': svm_accuracy,
                                'confusion_matrix': svm_cm,
                                'report': svm_report,
                                'y_pred': y_pred_svm,
                                'y_pred_proba': y_pred_proba_svm,
                                'training_time': time.time() - start_time
                            }
                        else:
                            # 创建和训练模型
                            if svm_kernel == "rbf":
                                svm_model = SVC(C=svm_C, kernel=svm_kernel, gamma=svm_gamma, probability=True, random_state=random_state)
                            else:
                                svm_model = SVC(C=svm_C, kernel=svm_kernel, probability=True, random_state=random_state)
                            
                            svm_model.fit(X_train_scaled, y_train)
                            
                            # 模型预测
                            y_pred_svm = svm_model.predict(X_test_scaled)
                            y_pred_proba_svm = svm_model.predict_proba(X_test_scaled)[:, 1]
                            
                            # 计算性能指标
                            svm_accuracy = accuracy_score(y_test, y_pred_svm)
                            svm_cm = confusion_matrix(y_test, y_pred_svm)
                            svm_report = classification_report(y_test, y_pred_svm)
                            
                            # 保存结果
                            results['SVM'] = {
                                'model': svm_model,
                                'accuracy': svm_accuracy,
                                'confusion_matrix': svm_cm,
                                'report': svm_report,
                                'y_pred': y_pred_svm,
                                'y_pred_proba': y_pred_proba_svm,
                                'training_time': time.time() - start_time
                            }
                    
                    progress_bar.progress(100)
                
                # 显示结果
                st.markdown("## 模型评估结果")
                
                # 为每个模型显示结果
                for model_name, model_results in results.items():
                    st.subheader(f"{model_name}模型结果")
                    
                    # 显示最佳参数（如果使用了网格搜索）
                    if use_grid_search and 'params' in model_results:
                        st.markdown(f"**最佳参数**: {model_results['params']}")
                    
                    # 显示性能指标
                    st.markdown(f"**准确率**: {model_results['accuracy']:.4f}")
                    st.markdown(f"**训练时间**: {model_results['training_time']:.4f} 秒")
                    
                    # 显示混淆矩阵
                    st.markdown("**混淆矩阵**:")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(model_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
                                xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
                    plt.ylabel('真实标签')
                    plt.xlabel('预测标签')
                    plt.title(f'{model_name}混淆矩阵')
                    st.pyplot(fig)
                    
                    # 显示分类报告
                    st.markdown("**分类报告**:")
                    st.text(model_results['report'])
                    
                    # 显示ROC曲线
                    st.markdown("**ROC曲线**:")
                    fpr, tpr, _ = roc_curve(y_test, model_results['y_pred_proba'])
                    roc_auc = auc(fpr, tpr)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.set_xlabel('假正例率 (FPR)')
                    ax.set_ylabel('真正例率 (TPR)')
                    ax.set_title(f'{model_name} ROC曲线')
                    ax.legend(loc='lower right')
                    st.pyplot(fig)
                
                # 如果是模型比较，显示比较结果
                if model_type == "模型比较" and len(results) > 1:
                    st.subheader("模型比较")
                    
                    # 比较准确率和训练时间
                    comparison_data = pd.DataFrame({
                        '模型': list(results.keys()),
                        '准确率': [results[model]['accuracy'] for model in results],
                        '训练时间(秒)': [results[model]['training_time'] for model in results]
                    })
                    st.table(comparison_data)
                    
                    # 比较ROC曲线
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    for model_name, model_results in results.items():
                        fpr, tpr, _ = roc_curve(y_test, model_results['y_pred_proba'])
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
                    
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.set_xlabel('假正例率 (FPR)')
                    ax.set_ylabel('真正例率 (TPR)')
                    ax.set_title('ROC曲线比较')
                    ax.legend(loc='lower right')
                    st.pyplot(fig)
            
            # 讨论和思考问题
            st.markdown("### 讨论和思考问题")
            
            st.markdown("""
            1. 数据集中存在哪些类型的特征？它们需要不同的预处理方法吗？
            
            2. 数据集中的类别不平衡会对模型性能产生什么影响？如何解决这个问题？
            
            3. 哪些特征对预测收入最重要？这些特征与现实世界的经济模式有何关联？
            
            4. 逻辑回归和SVM在这个问题上的表现有何不同？什么因素导致了这些差异？
            
            5. 如何解释模型的决策？哪种模型更容易解释？
            
            6. 如何进一步提高模型性能？考虑特征工程、参数调优和集成方法。
            """)

def show_advanced_exercises():
    """显示综合练习页面"""
    
    st.header("综合练习")
    
    st.markdown("""
    本节包含更复杂的综合练习，旨在将逻辑回归和SVM应用于真实世界的数据集和问题。
    这些练习将帮助你加深对分类算法的理解，并学习如何处理实际应用中的各种挑战。
    """)
    
    # 选择练习
    exercise = st.radio(
        "选择一个练习:",
        ["任务1: 收入预测"]
    )
    
    # 显示选定的练习
    if exercise == "任务1: 收入预测":
        show_income_prediction() 