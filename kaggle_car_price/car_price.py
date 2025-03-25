# 读取数据
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('kaggle_car_price/train.csv')

data.columns
# 分类型变量
categorical_features = ['symboling', 'CarName', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem']

# 数值型变量
numerical_features = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg']  

# 目标变量
target = 'price'

# 提取 X与 y
X = data[categorical_features + numerical_features]
y = data[target]


# 构造弹性网络模型, 使用5折交叉验证选择最佳超残

# 定义弹性网络模型, 首先使用 workflow 将分类型自变量转为哑变量

model = make_pipeline(
    ColumnTransformer(
        [('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
        remainder='passthrough'
    ),
    ElasticNet(alpha=0.05, l1_ratio=0.9)
)

model.fit(X, y)

# 在训练集上计算 RMSE
rmse = np.sqrt(mean_squared_error(y, model.predict(X)))
print(f"训练集上的 RMSE: {rmse:.2f}")

# 读取测试集
test_data = pd.read_csv('kaggle_car_price/test.csv')
# 计算测试集上的预测价格, 将结果保存为 csv 文件, 第一列为  car_ID, 第二列为  PredictedPrice
test_data['PredictedPrice'] = model.predict(test_data[categorical_features + numerical_features])
test_data[['car_ID', 'PredictedPrice']].to_csv('kaggle_car_price/submission.csv', index=False)

