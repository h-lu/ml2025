import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import sys
import os

# 导入matplotlib中文字体支持
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.plot_utils import configure_matplotlib_fonts

# 配置matplotlib支持中文
configure_matplotlib_fonts()

def show():
    st.title("XGBoost练习题")
    
    st.markdown("""
    本页面包含了一些用于巩固XGBoost知识的练习题，包括基础理论题和编程实践题。
    完成这些练习将帮助您更好地理解XGBoost算法及其应用。
    """)
    
    # 选择练习类型
    exercise_type = st.sidebar.radio(
        "练习类型",
        ["基础概念题", "参数理解练习", "编程实践", "扩展挑战"]
    )
    
    if exercise_type == "基础概念题":
        show_theory_exercises()
    elif exercise_type == "参数理解练习":
        show_parameter_exercises()
    elif exercise_type == "编程实践":
        show_coding_exercises()
    elif exercise_type == "扩展挑战":
        show_advanced_exercises()

def show_theory_exercises():
    st.header("基础概念题")
    
    st.markdown("""
    以下是一些关于集成学习和XGBoost的基础概念题，通过回答这些问题可以巩固您对相关理论的理解。
    """)
    
    # 练习1
    with st.expander("练习1：集成学习的分类", expanded=True):
        st.markdown("""
        **问题**：请将以下算法按照它们所属的集成学习类型(Bagging, Boosting, Stacking)分类：
        
        1. 随机森林 (Random Forest)
        2. AdaBoost
        3. Gradient Boosting
        4. XGBoost
        5. Voting Classifier
        """)
        
        # 用户回答
        answers = {
            "随机森林 (Random Forest)": st.selectbox("随机森林属于", ["选择答案", "Bagging", "Boosting", "Stacking"]),
            "AdaBoost": st.selectbox("AdaBoost属于", ["选择答案", "Bagging", "Boosting", "Stacking"]),
            "Gradient Boosting": st.selectbox("Gradient Boosting属于", ["选择答案", "Bagging", "Boosting", "Stacking"]),
            "XGBoost": st.selectbox("XGBoost属于", ["选择答案", "Bagging", "Boosting", "Stacking"]),
            "Voting Classifier": st.selectbox("Voting Classifier属于", ["选择答案", "Bagging", "Boosting", "Stacking"])
        }
        
        # 检查答案
        correct_answers = {
            "随机森林 (Random Forest)": "Bagging",
            "AdaBoost": "Boosting",
            "Gradient Boosting": "Boosting",
            "XGBoost": "Boosting",
            "Voting Classifier": "Stacking"
        }
        
        if st.button("提交答案", key="submit_ex1"):
            score = sum([1 for k, v in answers.items() if v == correct_answers[k]])
            st.metric("得分", f"{score}/5")
            
            for k, v in answers.items():
                if v == correct_answers[k]:
                    st.success(f"{k}: {v} ✓")
                elif v != "选择答案":
                    st.error(f"{k}: {v} ✗ (正确答案: {correct_answers[k]})")
    
    # 练习2
    with st.expander("练习2：XGBoost vs GBDT", expanded=False):
        st.markdown("""
        **问题**：XGBoost相比传统GBDT有哪些改进？请选择所有正确的选项：
        """)
        
        improvements = {
            "使用二阶导数进行优化": st.checkbox("使用二阶导数进行优化"),
            "内置正则化以减少过拟合": st.checkbox("内置正则化以减少过拟合"),
            "系统优化提高计算效率": st.checkbox("系统优化提高计算效率"),
            "支持并行计算": st.checkbox("支持并行计算"),
            "自动处理缺失值": st.checkbox("自动处理缺失值"),
            "减少了特征重要性评估能力": st.checkbox("减少了特征重要性评估能力"),
            "需要更少的训练样本": st.checkbox("需要更少的训练样本"),
            "支持列抽样": st.checkbox("支持列抽样")
        }
        
        correct_improvements = [
            "使用二阶导数进行优化",
            "内置正则化以减少过拟合",
            "系统优化提高计算效率",
            "支持并行计算",
            "自动处理缺失值",
            "支持列抽样"
        ]
        
        if st.button("提交答案", key="submit_ex2"):
            user_selections = [k for k, v in improvements.items() if v]
            correct_count = sum([1 for item in user_selections if item in correct_improvements])
            incorrect_count = len(user_selections) - correct_count
            
            # 计算得分：正确选项得1分，错误选项扣1分，最低0分
            score = max(0, correct_count - incorrect_count)
            total = len(correct_improvements)
            
            st.metric("得分", f"{score}/{total}")
            
            if score == total:
                st.success("全部正确！您对XGBoost的改进有很好的理解。")
            else:
                st.error("有些不正确。请复习XGBoost相对于传统GBDT的改进。")
                st.info(f"正确答案是: {', '.join(correct_improvements)}")
    
    # 练习3
    with st.expander("练习3：XGBoost的目标函数", expanded=False):
        st.markdown("""
        **问题**：XGBoost的目标函数由哪两部分组成？解释每部分的作用。
        """)
        
        user_answer = st.text_area("您的回答", height=150)
        
        reference_answer = """
        XGBoost的目标函数由两部分组成：
        
        1. 训练损失函数(Training Loss)：衡量模型对训练数据的拟合程度，常见的有均方误差(回归)、对数损失(分类)等。
        
        2. 正则化项(Regularization)：控制模型复杂度，防止过拟合。XGBoost使用的正则化包括树的叶子节点数量和叶子权重的L2范数。
        
        数学表示为：Obj = ∑L(yi, ŷi) + ∑Ω(fk)，其中L是损失函数，Ω是正则化项。
        """
        
        if st.button("查看参考答案", key="view_ex3"):
            st.info(reference_answer)
            
            if user_answer:
                st.write("您的回答：")
                st.write(user_answer)
                st.write("请与参考答案比较，自我评估您的理解。")

def show_parameter_exercises():
    st.header("参数理解练习")
    
    st.markdown("""
    以下练习旨在帮助您理解XGBoost的各种参数及其对模型性能的影响。
    """)
    
    # 参数匹配练习
    with st.expander("练习1：参数匹配", expanded=True):
        st.markdown("""
        **问题**：请将以下XGBoost参数与其功能/影响匹配起来：
        """)
        
        parameters = [
            "max_depth",
            "learning_rate",
            "n_estimators",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "gamma",
            "reg_alpha",
            "reg_lambda"
        ]
        
        descriptions = [
            "控制树的最大深度，增加可能导致过拟合",
            "学习率，控制每棵树对最终预测的贡献",
            "决策树的数量",
            "子节点中所需的最小权重和，用于控制过拟合",
            "用于训练每棵树的样本比例",
            "构建每棵树时随机抽样的特征比例",
            "节点分裂所需的最小损失减少值",
            "L1正则化参数",
            "L2正则化参数"
        ]
        
        # 打乱描述顺序
        np.random.seed(42)
        shuffled_desc = descriptions.copy()
        np.random.shuffle(shuffled_desc)
        
        # 用户选择
        user_matches = {}
        for param in parameters:
            user_matches[param] = st.selectbox(
                f"'{param}' 的功能是:",
                ["请选择"] + shuffled_desc
            )
        
        # 创建正确答案字典
        correct_matches = dict(zip(parameters, descriptions))
        
        if st.button("检查答案", key="check_param_match"):
            correct_count = sum([1 for param, desc in user_matches.items() 
                               if desc == correct_matches[param]])
            
            st.metric("正确匹配数", f"{correct_count}/{len(parameters)}")
            
            if correct_count == len(parameters):
                st.success("全部正确！您对XGBoost参数有很好的理解。")
                st.balloons()
            else:
                for param, desc in user_matches.items():
                    if desc == correct_matches[param]:
                        st.success(f"✓ {param}: {desc}")
                    elif desc != "请选择":
                        st.error(f"✗ {param}: 您选择了 '{desc}'，正确答案是 '{correct_matches[param]}'")
    
    # 参数效果预测
    with st.expander("练习2：参数效果预测", expanded=False):
        st.markdown("""
        **问题**：预测以下参数变化对模型性能的影响：
        """)
        
        scenarios = {
            "增加max_depth": st.radio(
                "增加max_depth可能导致：",
                ["过拟合风险增加", "欠拟合风险增加", "无明显影响"],
                key="max_depth_effect"
            ),
            "减小learning_rate同时增加n_estimators": st.radio(
                "减小learning_rate同时增加n_estimators可能导致：",
                ["模型性能下降", "模型性能提升", "训练速度提升但性能不变"],
                key="lr_nestimators_effect"
            ),
            "增加subsample和colsample_bytree": st.radio(
                "增加subsample和colsample_bytree可能导致：",
                ["减少过拟合", "增加过拟合", "提高训练速度但可能增加过拟合"],
                key="sampling_effect"
            ),
            "增加reg_alpha和reg_lambda": st.radio(
                "增加reg_alpha和reg_lambda可能导致：",
                ["模型更复杂", "模型更简单", "训练速度更快"],
                key="reg_effect"
            )
        }
        
        correct_answers = {
            "增加max_depth": "过拟合风险增加",
            "减小learning_rate同时增加n_estimators": "模型性能提升",
            "增加subsample和colsample_bytree": "提高训练速度但可能增加过拟合",
            "增加reg_alpha和reg_lambda": "模型更简单"
        }
        
        if st.button("提交答案", key="submit_param_effects"):
            score = sum([1 for k, v in scenarios.items() if v == correct_answers[k]])
            st.metric("得分", f"{score}/{len(scenarios)}")
            
            for scenario, answer in scenarios.items():
                if answer == correct_answers[scenario]:
                    st.success(f"{scenario}: {answer} ✓")
                else:
                    st.error(f"{scenario}: {answer} ✗ (正确答案: {correct_answers[scenario]})")
            
            explanations = {
                "增加max_depth": "增加树的深度会使模型更复杂，能捕获更多的模式，但也更容易拟合训练数据中的噪声，导致过拟合。",
                "减小learning_rate同时增加n_estimators": "这通常被称为'慢学习'策略，可以让模型更稳定地学习，通常能提高性能，但会增加训练时间。",
                "增加subsample和colsample_bytree": "增加这些参数接近1.0会减少随机性，可能会提高训练速度，但更可能导致过拟合，因为使用了更多的数据。",
                "增加reg_alpha和reg_lambda": "增加正则化参数会惩罚复杂模型，使模型倾向于更简单的结构，有助于减少过拟合。"
            }
            
            st.markdown("### 解释")
            for scenario, explanation in explanations.items():
                st.info(f"{scenario}: {explanation}")

def show_coding_exercises():
    st.header("编程实践")
    
    st.markdown("""
    以下是一些XGBoost的编程练习，通过实际编写代码来巩固您的理解。
    每个练习都有提示和参考解答。
    """)
    
    # 练习1：基本模型训练
    with st.expander("练习1：基本XGBoost模型训练", expanded=True):
        st.markdown("""
        **任务**：完成以下代码，训练一个XGBoost回归模型并评估其性能。
        
        使用以下生成的非线性数据：
        ```python
        # 生成数据
        np.random.seed(42)
        X = np.random.rand(1000, 5)
        y = 5 + 3*X[:, 0]**2 + 2*X[:, 1] + np.sin(3*X[:, 2]) + np.exp(X[:, 3]) - 2*X[:, 4] + np.random.normal(0, 1, 1000)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        ```
        
        您需要:
        1. 创建并训练XGBoost模型
        2. 进行预测
        3. 计算RMSE和R²性能指标
        """)
        
        # 代码编辑器
        user_code = st.text_area(
            "在这里编写您的代码:",
            """# 已有的代码
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# 生成数据
np.random.seed(42)
X = np.random.rand(1000, 5)
y = 5 + 3*X[:, 0]**2 + 2*X[:, 1] + np.sin(3*X[:, 2]) + np.exp(X[:, 3]) - 2*X[:, 4] + np.random.normal(0, 1, 1000)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 您的代码开始
# 1. 创建XGBoost模型

# 2. 训练模型

# 3. 预测

# 4. 计算评估指标

# 您的代码结束
""",
            height=350
        )
        
        # 提示
        if st.button("显示提示", key="hint_ex1"):
            st.info("""
            提示：
            1. 使用 `xgb.XGBRegressor()` 创建模型，设置适当的参数
            2. 使用 `model.fit(X_train, y_train)` 训练模型
            3. 使用 `model.predict(X_test)` 进行预测
            4. 使用 `np.sqrt(mean_squared_error(y_test, y_pred))` 计算RMSE
            5. 使用 `r2_score(y_test, y_pred)` 计算R²
            """)
        
        # 参考答案
        if st.button("显示参考答案", key="solution_ex1"):
            st.code("""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# 生成数据
np.random.seed(42)
X = np.random.rand(1000, 5)
y = 5 + 3*X[:, 0]**2 + 2*X[:, 1] + np.sin(3*X[:, 2]) + np.exp(X[:, 3]) - 2*X[:, 4] + np.random.normal(0, 1, 1000)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建XGBoost模型
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算评估指标
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
            """)
    
    # 练习2：交叉验证
    with st.expander("练习2：模型交叉验证", expanded=False):
        st.markdown("""
        **任务**：完成以下代码，使用交叉验证来评估XGBoost模型的性能，并计算平均RMSE。
        
        继续使用练习1中的数据。
        """)
        
        # 代码编辑器
        user_code = st.text_area(
            "在这里编写您的代码:",
            """# 已有的代码
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
import xgboost as xgb

# 生成数据
np.random.seed(42)
X = np.random.rand(1000, 5)
y = 5 + 3*X[:, 0]**2 + 2*X[:, 1] + np.sin(3*X[:, 2]) + np.exp(X[:, 3]) - 2*X[:, 4] + np.random.normal(0, 1, 1000)

# 您的代码开始
# 1. 创建XGBoost模型

# 2. 定义RMSE评分器

# 3. 执行交叉验证

# 4. 计算平均RMSE

# 您的代码结束
""",
            height=300,
            key="code_cv"
        )
        
        # 提示
        if st.button("显示提示", key="hint_ex2"):
            st.info("""
            提示：
            1. 创建XGBoost模型与练习1相同
            2. 使用 `make_scorer(mean_squared_error, greater_is_better=False, squared=False)` 创建RMSE评分器
            3. 使用 `cross_val_score()` 函数进行交叉验证，指定scoring参数
            4. 计算交叉验证分数的均值
            """)
        
        # 参考答案
        if st.button("显示参考答案", key="solution_ex2"):
            st.code("""
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
import xgboost as xgb

# 生成数据
np.random.seed(42)
X = np.random.rand(1000, 5)
y = 5 + 3*X[:, 0]**2 + 2*X[:, 1] + np.sin(3*X[:, 2]) + np.exp(X[:, 3]) - 2*X[:, 4] + np.random.normal(0, 1, 1000)

# 创建XGBoost模型
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

# 定义RMSE评分器
rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)

# 执行交叉验证
cv_scores = cross_val_score(
    model, 
    X, 
    y, 
    cv=5, 
    scoring=rmse_scorer
)

# 计算平均RMSE
mean_rmse = -np.mean(cv_scores)  # 负号是因为make_scorer使用greater_is_better=False
print(f"交叉验证平均RMSE: {mean_rmse:.4f}")
            """)

def show_advanced_exercises():
    st.header("扩展挑战")
    
    st.markdown("""
    以下是一些更具挑战性的练习，旨在测试您对XGBoost的深入理解和应用能力。
    """)
    
    # 挑战1：参数调优
    with st.expander("挑战1：网格搜索参数调优", expanded=True):
        st.markdown("""
        **任务**：使用网格搜索找到XGBoost模型的最佳参数组合。
        
        要求：
        1. 使用GridSearchCV或RandomizedSearchCV进行参数搜索
        2. 至少调整以下参数：max_depth, learning_rate, n_estimators, min_child_weight
        3. 使用5折交叉验证
        4. 报告最佳参数组合和对应的性能指标
        """)
        
        st.markdown("""
        参考代码框架:
        ```python
        from sklearn.model_selection import GridSearchCV
        
        # 创建参数网格
        param_grid = {
            'max_depth': [...],
            'learning_rate': [...],
            'n_estimators': [...],
            'min_child_weight': [...]
        }
        
        # 创建模型
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        
        # 创建网格搜索
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=5,
            verbose=1,
            n_jobs=-1
        )
        
        # 执行网格搜索
        grid_search.fit(X_train, y_train)
        
        # 获取最佳参数
        best_params = grid_search.best_params_
        
        # 获取最佳模型性能
        best_score = np.sqrt(-grid_search.best_score_)
        ```
        """)
        
        # 提交代码按钮
        st.markdown("### 提交您的解决方案")
        
        with st.form("submit_challenge1"):
            user_solution = st.text_area(
                "粘贴您的代码解决方案:",
                height=300
            )
            
            submitted = st.form_submit_button("提交")
            
            if submitted and user_solution:
                st.success("代码已提交！在实际应用中，这里会评估您的代码并提供反馈。")
    
    # 挑战2：特征重要性分析
    with st.expander("挑战2：特征重要性分析与可视化", expanded=False):
        st.markdown("""
        **任务**：训练XGBoost模型并分析特征重要性，使用多种可视化方法展示结果。
        
        要求：
        1. 使用波士顿房价数据集或其他合适的数据集
        2. 训练XGBoost模型并获取特征重要性
        3. 使用至少两种不同方法可视化特征重要性(如条形图和SHAP值)
        4. 解释哪些特征对预测最重要，为什么
        """)
        
        st.markdown("""
        参考代码框架:
        ```python
        from sklearn.datasets import load_boston
        import matplotlib.pyplot as plt
        import shap  # 需要安装shap库
        
        # 加载数据
        boston = load_boston()
        X, y = boston.data, boston.target
        feature_names = boston.feature_names
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 训练模型
        model = xgb.XGBRegressor(...)
        model.fit(X_train, y_train)
        
        # 方法1：内置特征重要性可视化
        xgb.plot_importance(model)
        plt.title('特征重要性(基于权重)')
        plt.tight_layout()
        plt.show()
        
        # 方法2：使用SHAP值
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # SHAP摘要图
        shap.summary_plot(shap_values, X_test, feature_names=feature_names)
        ```
        """)
        
        # 提交代码按钮
        st.markdown("### 提交您的解决方案")
        
        with st.form("submit_challenge2"):
            user_solution = st.text_area(
                "粘贴您的代码解决方案:",
                height=300,
                key="solution_ch2"
            )
            
            user_explanation = st.text_area(
                "您对特征重要性的分析和解释:",
                height=150
            )
            
            submitted = st.form_submit_button("提交")
            
            if submitted and user_solution:
                st.success("解决方案已提交！在实际应用中，这里会评估您的代码、分析和可视化效果。")
                
                if user_explanation:
                    st.info("您提供的解释：")
                    st.write(user_explanation)
                    st.write("在实际应用中，这里会对您的解释给予反馈。") 