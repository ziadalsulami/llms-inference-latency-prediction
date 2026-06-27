from datasets import load_dataset
import pandas as pd
from sklearn import linear_model
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ds = load_dataset('rbgo/llm-inference-benchmark')


split = ds['train'].train_test_split(test_size=0.2, seed=42)

df_train = split['train'].to_pandas()
df_test = split['test'].to_pandas()

features = ['Token_Count', 'output_length', 'input_length']

x_train = df_train[features]
y_train = df_train['Latency']

x_test = df_test[features]
y_test = df_test['Latency']

reg = linear_model.LinearRegression()
reg.fit(x_train,y_train)

y_pred=reg.predict(x_test)
print("-------LINEAR REGRESSION RESULTS-----")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("-------RANDOM FOREST RESULTS-----")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")