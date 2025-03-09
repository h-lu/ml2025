# 第四周：分类算法基础（二）- 决策树与集成学习（随机森林）

这是一个使用Streamlit构建的交互式课件，详细展示了决策树和随机森林算法的原理与应用。

## 功能特点

1. **模块化设计**: 应用拆分为多个相互独立的模块，使得每个文件都清晰简洁，容易维护。

2. **主题涵盖全面**:
   - 决策树算法的原理、特征选择方法和实现
   - 集成学习和随机森林的概念和应用
   - 模型评估与调优方法
   - 基础和高级练习部分

3. **交互式学习**:
   - 用户可以通过调整参数实时看到结果变化
   - 可视化帮助理解算法原理和结果
   - 练习部分包含完整参考代码和思考问题

4. **高级练习模块**:
   - 集成学习方法比较：比较随机森林、Gradient Boosting和AdaBoost
   - 特征重要性分析：深入理解不同特征重要性计算方法
   - 超参数调优：使用网格搜索和学习曲线进行模型调优

## 安装与运行

1. 安装依赖包:

```bash
pip install -r requirements.txt
```

2. 运行应用:

```bash
# 方法1：使用run.py脚本
python run.py

# 方法2：直接使用streamlit命令
streamlit run app.py
```

## 文件结构

```
week4_app/
├── app.py                    # 主应用入口
├── run.py                    # 运行脚本
├── requirements.txt          # 依赖包列表
├── README.md                 # 项目说明
├── pages/                    # 页面模块
│   ├── __init__.py  
│   ├── decision_tree.py      # 决策树内容
│   ├── ensemble.py           # 集成学习和随机森林
│   ├── evaluation.py         # 模型评估与选择
│   ├── basic_exercises.py    # 基础练习
│   └── advanced_exercises.py # 高级练习的导航页
├── exercises/                # 高级练习模块
│   ├── __init__.py
│   ├── ensemble_comparison.py # 集成学习方法比较
│   ├── feature_importance.py  # 特征重要性分析
│   └── model_tuning.py        # 超参数调优
├── utils/                     # 工具函数
│   ├── __init__.py
│   ├── visualization.py       # 可视化工具
│   ├── data_loader.py         # 数据加载工具
│   └── helpers.py             # 辅助函数
└── img/                       # SVG图像资源
    ├── decision_tree_concept.svg    # 决策树概念图
    ├── random_forest_concept.svg    # 随机森林概念图
    ├── feature_importance_concept.svg  # 特征重要性概念图
    └── cross_validation_concept.svg    # 交叉验证概念图
```

## 系统要求

- Python 3.7+
- 现代浏览器

## 主要依赖包

- streamlit==1.32.0
- pandas==2.1.0
- numpy==1.24.3
- scikit-learn==1.3.2
- matplotlib==3.7.2
- seaborn==0.12.2
- plotly==5.18.0
- graphviz==0.20.1 