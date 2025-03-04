import streamlit as st
from utils.assessment import create_quiz

def show_logistic_regression_quiz():
    """显示逻辑回归测验"""
    
    questions = [
        {
            "question": "逻辑回归的输出是什么？",
            "options": [
                "连续值，如同线性回归",
                "概率值，范围在[0,1]之间",
                "只有0或1的二元值",
                "任意范围的实数值"
            ],
            "correct_index": 1,
            "explanation": "逻辑回归通过Sigmoid函数将线性组合输出转换为概率值，范围在[0,1]之间。"
        },
        {
            "question": "逻辑回归中常用的损失函数是什么？",
            "options": [
                "均方误差(MSE)",
                "绝对误差(MAE)",
                "交叉熵损失(Cross Entropy Loss)",
                "Hinge Loss"
            ],
            "correct_index": 2,
            "explanation": "逻辑回归通常使用交叉熵损失函数，它对概率预测的偏差非常敏感。"
        },
        {
            "question": "逻辑回归模型中的sigmoid函数表达式是什么？",
            "options": [
                "σ(z) = 1/(1-e^(-z))",
                "σ(z) = 1/(1+e^(-z))",
                "σ(z) = e^z/(1+e^z)",
                "σ(z) = z/(1+|z|)"
            ],
            "correct_index": 1,
            "explanation": "Sigmoid函数的标准表达式是σ(z) = 1/(1+e^(-z))，将任意实数映射到(0,1)区间。"
        },
        {
            "question": "在逻辑回归中，如何处理过拟合问题？",
            "options": [
                "增加模型的复杂度",
                "收集更多数据",
                "添加L1或L2正则化",
                "使用更复杂的激活函数"
            ],
            "correct_index": 2,
            "explanation": "添加L1(Lasso)或L2(Ridge)正则化是处理逻辑回归过拟合的常用方法，通过惩罚大权重来减少模型复杂度。"
        },
        {
            "question": "逻辑回归模型的决策边界是什么形状？",
            "options": [
                "总是直线",
                "总是曲线",
                "在原始特征空间中是线性的，但可以通过特征工程变成非线性的",
                "随机形状"
            ],
            "correct_index": 2,
            "explanation": "在原始特征空间中，逻辑回归的决策边界是线性的。但通过多项式特征等特征工程，可以创建非线性决策边界。"
        }
    ]
    
    return create_quiz("逻辑回归基础", questions)

def show_svm_quiz():
    """显示SVM测验"""
    
    questions = [
        {
            "question": "支持向量机(SVM)的主要目标是什么？",
            "options": [
                "最小化训练误差",
                "找到将数据分成两类的任意超平面",
                "找到具有最大间隔的决策边界",
                "最小化支持向量的数量"
            ],
            "correct_index": 2,
            "explanation": "SVM的核心目标是找到能够以最大间隔分隔两个类别的决策边界，这有助于提高模型的泛化能力。"
        },
        {
            "question": "在SVM中，什么是'支持向量'？",
            "options": [
                "所有训练样本",
                "最靠近决策边界的数据点",
                "模型的权重向量",
                "核函数的参数"
            ],
            "correct_index": 1,
            "explanation": "支持向量是那些最靠近决策边界的数据点，它们'支持'或确定了最优决策边界的位置。"
        },
        {
            "question": "SVM中的软间隔(soft margin)允许什么？",
            "options": [
                "更小的间隔",
                "非线性决策边界",
                "某些训练样本的错误分类",
                "更快的训练速度"
            ],
            "correct_index": 2,
            "explanation": "软间隔允许一些样本被错误分类或落在间隔内，通过引入松弛变量和惩罚参数C来平衡间隔大小和错误分类。"
        },
        {
            "question": "在SVM中，核函数(kernel function)的作用是什么？",
            "options": [
                "加速模型训练",
                "减少模型复杂度",
                "在高维空间计算点积，而无需显式转换",
                "减少支持向量的数量"
            ],
            "correct_index": 2,
            "explanation": "核函数让SVM能够在高维空间中隐式计算点积，而无需显式转换特征，从而使SVM可以处理非线性问题。"
        },
        {
            "question": "SVM中参数C的作用是什么？",
            "options": [
                "控制学习率",
                "控制正则化强度",
                "控制特征数量",
                "控制支持向量数量"
            ],
            "correct_index": 1,
            "explanation": "参数C控制正则化强度，较大的C值会让模型更注重减少误分类，可能导致过拟合；较小的C值则更强调较大的间隔，可能导致欠拟合。"
        }
    ]
    
    return create_quiz("SVM基础", questions)

def show_model_comparison_quiz():
    """显示模型比较测验"""
    
    questions = [
        {
            "question": "逻辑回归与SVM的主要区别是什么？",
            "options": [
                "逻辑回归只能处理二分类问题，SVM可以处理多分类",
                "逻辑回归输出概率，SVM直接输出类别",
                "逻辑回归总是线性的，SVM总是非线性的",
                "逻辑回归不能处理高维数据，SVM可以"
            ],
            "correct_index": 1,
            "explanation": "一个关键区别是逻辑回归输出概率值，而SVM直接输出类别或距离决策边界的距离。两者都可以扩展到多分类问题。"
        },
        {
            "question": "在小样本、高维特征的情况下，通常哪种算法表现更好？",
            "options": [
                "逻辑回归",
                "支持向量机",
                "两者效果相同",
                "取决于数据分布"
            ],
            "correct_index": 1,
            "explanation": "在小样本、高维特征的情况下，SVM通常表现更好，因为它专注于边界样本(支持向量)而非所有样本，且有内置的正则化机制。"
        },
        {
            "question": "当需要知道预测的概率而非仅仅是类别时，应该选择哪个算法？",
            "options": [
                "逻辑回归",
                "支持向量机",
                "两者都可以",
                "需要其他算法"
            ],
            "correct_index": 0,
            "explanation": "逻辑回归天然输出概率值。虽然有方法可以将SVM输出转换为概率估计(如Platt scaling)，但逻辑回归在这方面更为直接。"
        },
        {
            "question": "关于计算复杂度，下列哪项是正确的？",
            "options": [
                "逻辑回归总是比SVM快",
                "SVM总是比逻辑回归快",
                "对于大数据集，逻辑回归通常比SVM更高效",
                "两者计算复杂度相同"
            ],
            "correct_index": 2,
            "explanation": "对于大数据集，逻辑回归通常比SVM更高效。SVM的训练复杂度随着样本数量的增加而显著增加，特别是使用非线性核函数时。"
        },
        {
            "question": "在处理非线性问题时，以下哪种方法是正确的？",
            "options": [
                "逻辑回归总是优于SVM",
                "SVM使用核技巧处理非线性，逻辑回归需要显式特征转换",
                "两种算法都不能处理非线性问题",
                "逻辑回归和SVM处理非线性问题的方法完全相同"
            ],
            "correct_index": 1,
            "explanation": "SVM可以通过核技巧隐式地在高维空间工作，而逻辑回归需要显式的特征转换(如添加多项式特征)来处理非线性问题。"
        }
    ]
    
    return create_quiz("模型比较", questions)

def show_practical_quiz():
    """显示实践应用测验"""
    
    questions = [
        {
            "question": "在文本分类任务中，以下哪个模型可能更适合处理高维稀疏特征？",
            "options": [
                "线性SVM",
                "带RBF核的SVM",
                "朴素贝叶斯",
                "决策树"
            ],
            "correct_index": 0,
            "explanation": "线性SVM在处理高维稀疏特征(如文本的词袋或TF-IDF表示)方面表现良好，计算效率也比非线性SVM高。"
        },
        {
            "question": "当处理不平衡数据集时，逻辑回归中可以采取的策略是？",
            "options": [
                "只使用多数类的样本",
                "调整决策阈值",
                "使用更复杂的模型",
                "总是使用默认阈值0.5"
            ],
            "correct_index": 1,
            "explanation": "在不平衡数据集中，调整决策阈值(不总是使用0.5)是处理逻辑回归输出的有效策略，可以根据不同的精确率/召回率需求来设置。"
        },
        {
            "question": "在医疗诊断等高风险应用中，模型选择时应该优先考虑什么？",
            "options": [
                "计算速度",
                "模型复杂度",
                "模型可解释性",
                "算法新颖性"
            ],
            "correct_index": 2,
            "explanation": "在医疗诊断等高风险领域，模型可解释性通常是优先考虑的因素，因为需要理解和验证模型的决策过程。这方面逻辑回归往往优于复杂的黑盒模型。"
        },
        {
            "question": "以下哪种情况下，增加特征可能会导致SVM性能下降？",
            "options": [
                "特征与目标无关",
                "特征之间高度相关",
                "特征数量超过样本数量",
                "以上所有情况"
            ],
            "correct_index": 3,
            "explanation": "所有这些情况都可能导致SVM性能下降：无关特征增加了噪声，相关特征导致冗余计算，特征数超过样本数可能导致过拟合。"
        },
        {
            "question": "在实际应用中评估分类模型时，以下哪个指标最不适合用于不平衡数据集？",
            "options": [
                "精确率(Precision)",
                "召回率(Recall)",
                "F1分数",
                "准确率(Accuracy)"
            ],
            "correct_index": 3,
            "explanation": "准确率在不平衡数据集上通常是有误导性的，因为简单地预测多数类就能获得较高的准确率。精确率、召回率和F1分数能更好地评估不平衡数据集上的模型性能。"
        }
    ]
    
    return create_quiz("实践应用", questions)

def show_coding_quiz():
    """显示编程实践测验"""
    
    questions = [
        {
            "question": "在scikit-learn中使用逻辑回归时，哪个参数用于控制正则化强度？",
            "options": [
                "alpha",
                "C",
                "lambda",
                "regularization"
            ],
            "correct_index": 1,
            "explanation": "在scikit-learn的LogisticRegression中，参数C控制正则化强度，C值越小，正则化越强。"
        },
        {
            "question": "在scikit-learn中，SVC类中的kernel参数默认值是什么？",
            "options": [
                "linear",
                "poly",
                "rbf",
                "sigmoid"
            ],
            "correct_index": 2,
            "explanation": "scikit-learn中SVC类的kernel参数默认值是'rbf'(径向基函数核)。"
        },
        {
            "question": "在使用逻辑回归进行多分类问题时，以下哪个参数设置是正确的？",
            "options": [
                "multi_class='multinomial', solver='liblinear'",
                "multi_class='ovr', solver='lbfgs'",
                "multi_class='multinomial', solver='lbfgs'",
                "multi_class='binary', solver='newton-cg'"
            ],
            "correct_index": 2,
            "explanation": "对于多分类问题，'multinomial'选项使用softmax函数，而'lbfgs'是支持multinomial选项的优化器。'liblinear'不支持multinomial选项。"
        },
        {
            "question": "在SVM中，以下哪个核函数最适合处理无限维特征空间？",
            "options": [
                "线性核(linear)",
                "多项式核(polynomial)",
                "RBF核(径向基函数)",
                "Sigmoid核"
            ],
            "correct_index": 2,
            "explanation": "RBF核(高斯核)可以映射到无限维特征空间，使其能够捕获非常复杂的决策边界。"
        },
        {
            "question": "当使用sklearn的Pipeline和GridSearchCV进行模型选择时，参数名称的正确格式是什么？",
            "options": [
                "step.parameter",
                "step_parameter",
                "step[parameter]",
                "step__parameter"
            ],
            "correct_index": 3,
            "explanation": "在sklearn的Pipeline中传递参数时，正确的格式是'step__parameter'，使用双下划线连接步骤名和参数名。"
        }
    ]
    
    return create_quiz("编程实践", questions)

def show_quizzes():
    """显示所有测验"""
    st.header("分类算法知识测验")
    
    st.markdown("""
    通过完成以下测验来检验你对逻辑回归和支持向量机的理解。每个测验包含5个问题，
    完成后将显示你的得分和正确答案解析。
    
    这些测验将帮助你：
    - 巩固对算法原理的理解
    - 识别知识盲点
    - 加深对实际应用场景的认识
    """)
    
    quiz_type = st.selectbox(
        "选择测验类型",
        ["逻辑回归基础", "SVM基础", "模型比较", "实践应用", "编程实践"]
    )
    
    if quiz_type == "逻辑回归基础":
        show_logistic_regression_quiz()
    elif quiz_type == "SVM基础":
        show_svm_quiz()
    elif quiz_type == "模型比较":
        show_model_comparison_quiz()
    elif quiz_type == "实践应用":
        show_practical_quiz()
    elif quiz_type == "编程实践":
        show_coding_quiz() 