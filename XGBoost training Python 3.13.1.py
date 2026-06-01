# ==================== 第一部分：模型训练与调优 ====================
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Continuous, Categorical, Integer
import pickle

# 数据加载与预处理
data = pd.read_excel('D:\\Case studies\\Code\\ANN training modified.xlsx')
X = data.iloc[:, 0:12].values
#y = data.iloc[:, 12].values
y = np.round(data.iloc[:, 14].values, 3)  # 原始目标值保留3位小数

# 数据归一化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义XGBRegressor参数
model = XGBRegressor(
    objective='reg:squarederror',
    booster='gbtree',  # 切换为dart树类型
    tree_method='hist',  # 使用直方图优化
    random_state=42,
    enable_categorical=False
)
# 定义参数搜索空间
param_grid = {
    'max_depth': Integer(3, 15),
    'learning_rate': Continuous(0.005, 0.3),
    'n_estimators': Integer(500, 5000),
    'subsample': Continuous(0.5, 1.0),
    'colsample_bytree': Continuous(0.5, 1.0),
    'gamma': Continuous(0, 0.3),
    'reg_alpha': Continuous(0, 1),
    'reg_lambda': Continuous(0.5, 2),
    'grow_policy': Categorical(['depthwise', 'lossguide'])
}

# 遗传算法优化
genetic_search = GASearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    population_size=30,
    generations=300,
    crossover_probability=0.8,  # 调高交叉概率
    mutation_probability=0.05,  # 降低变异概率
    n_jobs=-1,
    verbose=True
)

genetic_search.fit(X_train, y_train)

# 模型评估与保存
best_model = genetic_search.best_estimator_
#y_pred = best_model.predict(X_test)
y_pred = np.round(best_model.predict(X_test), 3)  # 预测值保留3位小数

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'最优参数: {genetic_search.best_params_}')
print(f'测试集 MSE: {mse:.6f}')
print(f'测试集 R²: {r2:.6f}')

# 特征重要性分析
importance = best_model.feature_importances_
for i,v in enumerate(importance):
    print(f'Feature {i}: Score {v:.5f}')

# 保存模型和标准化器
model.save_model("xgboost_model_UDI.json")  # 保存模型本体
with open("scaler_UDI.pkl", "wb") as f:    # 单独保存scaler
    pickle.dump(scaler, f)
