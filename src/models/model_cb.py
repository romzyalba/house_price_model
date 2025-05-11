import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostRegressor

# Splitting dataset
rumah123_df = pd.read_csv("../../data/rumah123_ready.csv")
X = rumah123_df.drop(columns=['price'])
y = rumah123_df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fitur Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modeling
best_params = {'bagging_temperature': 0.1, 'random_strength': 1, 
               'depth': 7, 'learning_rate': 0.15, 
               'l2_leaf_reg': 2, 'iterations': 1500}
catboost_grid_best = CatBoostRegressor(**best_params, logging_level='Silent')
catboost_grid_best.fit(X_train_scaled, y_train)

joblib.dump(catboost_grid_best, '../../models/model_cb.joblib')