# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
# 假设线性回归数据已准备好
np.random.seed(0)
X_lin = 2 * np.random.rand(100, 1)
y_lin = 4 + 3 * X_lin + np.random.randn(100, 1)
X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(
    X_lin, y_lin, test_size=0.2, random_state=42
)


# --- 假设已有 X_train, X_test, y_train, y_test (来自项目二预处理后的数据) ---
# 如果没有，需要重新加载数据并划分
# 示例：使用上周的线性数据 X_train_lin, X_test_lin, y_train_lin, y_test_lin
# 注意：XGBoost 对特征缩放不敏感，但如果之前做了缩放也没关系

# --- 训练 XGBoost 回归模型 ---
# 1. 创建模型实例 (常用参数解释见下文)
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', # 回归任务，目标是最小化平方误差
                           n_estimators=100,         # 树的数量 (基学习器数量)
                           learning_rate=0.1,        # 学习率 (步长)，控制每棵树的贡献，防止过拟合
                           max_depth=3,              # 每棵树的最大深度
                           subsample=0.8,            # 训练每棵树时随机抽取的样本比例
                           colsample_bytree=0.8,     # 训练每棵树时随机抽取的特征比例
                           gamma=0,                  # 控制是否后剪枝的参数 (越大越保守)
                           reg_alpha=0,              # L1 正则化项系数
                           reg_lambda=1,             # L2 正则化项系数 (默认)
                           random_state=42,
                           n_jobs=-1)

# 2. 拟合模型
# XGBoost 可以使用验证集进行早停 (Early Stopping) 来防止过拟合，提高效率
# eval_set: 提供一个或多个验证集用于评估
# early_stopping_rounds: 如果验证集上的评估指标连续 N 轮没有改善，则停止训练
eval_set = [(X_train_lin, y_train_lin), (X_test_lin, y_test_lin)]
xgb_reg.fit(X_train_lin, y_train_lin,
            eval_set=eval_set,
            eval_metric='rmse', # 指定在验证集上监控的指标
            early_stopping_rounds=10,
            verbose=False) # verbose=True 会打印每一轮的评估结果

# --- 进行预测 ---
y_pred_xgb = xgb_reg.predict(X_test_lin)

# --- 评估模型 ---
print("\n--- XGBoost 回归评估 ---")
mse_xgb = mean_squared_error(y_test_lin, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = r2_score(y_test_lin, y_pred_xgb)

print(f"XGBoost 均方根误差 (RMSE): {rmse_xgb:.4f}")
print(f"XGBoost R 方 (R-squared): {r2_xgb:.4f}")

# --- (可选) 对比线性回归结果 ---
# from sklearn.linear_model import LinearRegression # 需要导入
# lin_reg = LinearRegression().fit(X_train_lin, y_train_lin)
# y_pred_lin = lin_reg.predict(X_test_lin)
# rmse_lin = np.sqrt(mean_squared_error(y_test_lin, y_pred_lin))
# r2_lin = r2_score(y_test_lin, y_pred_lin)
# print(f"\n线性回归 RMSE: {rmse_lin:.4f}")
# print(f"线性回归 R 方: {r2_lin:.4f}")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
