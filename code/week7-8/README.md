# 无监督学习聚类算法演示

这是一个基于Streamlit的交互式应用，用于演示无监督学习中的K-Means和DBSCAN聚类算法。本应用包含多个模块，涵盖了第七周和第八周讲座的核心内容。

## 项目结构

- `app.py`: 主应用程序入口点，包含侧边栏导航和页面框架
- `data_generator.py`: 生成用于演示的各种聚类数据
- `kmeans_demo.py`: K-Means聚类算法演示模块
- `dbscan_demo.py`: DBSCAN聚类算法演示模块
- `clustering_comparison.py`: K-Means与DBSCAN算法对比模块
- `business_insights.py`: 聚类结果的业务解读与用户分群案例

## 功能模块

1. **简介与数据生成**: 无监督学习基本概念和生成不同形状的聚类数据
2. **K-Means聚类**: K-Means算法原理、K值选择方法(肘部法则、轮廓系数)及实现演示
3. **DBSCAN聚类**: DBSCAN算法原理、参数选择方法(K-距离图)及实现演示
4. **聚类算法比较**: 在不同形状数据上比较两种算法的效果
5. **业务洞察解读**: 用户分群案例与业务解读

## 运行方法

确保已安装所需依赖:

```bash
pip install streamlit numpy pandas matplotlib seaborn scikit-learn
```

在项目目录下运行:

```bash
streamlit run app.py
```

## 中文字体支持

应用已添加对中文字体的支持，根据不同操作系统自动选择合适的字体：

- **Mac系统**: 使用 Arial Unicode MS, PingFang SC, STHeiti 等字体
- **Windows系统**: 使用 Microsoft YaHei (微软雅黑), SimHei (黑体), SimSun (宋体) 等字体
- **Linux系统**: 使用 DejaVu Sans, WenQuanYi Micro Hei 等字体

如果图表中仍然出现中文乱码，请确保系统已安装上述字体。对于 Mac 系统，大部分字体已预装；对于 Windows 系统，请确保已安装中文语言包。

## 主要特点

- **交互式演示**: 调整各种参数实时查看效果
- **可视化比较**: 直观对比不同算法的聚类结果
- **算法选择指南**: 帮助理解何时选择哪种聚类算法
- **业务分析案例**: 如何从聚类结果中提取商业价值

## 学习目标

通过本演示应用，您将能够:

1. 理解K-Means和DBSCAN聚类算法的基本原理
2. 掌握如何选择合适的聚类算法参数
3. 学习聚类结果评估方法
4. 了解如何将聚类结果转化为商业洞察
5. 实践用户分群分析的完整流程 