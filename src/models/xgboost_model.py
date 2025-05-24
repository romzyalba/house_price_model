import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

# Load data
rumah123_df = pd.read_csv("../../data/rumah123_ready.csv")
X = rumah123_df.drop(columns=["price"])
y = rumah123_df["price"]

# Parameter hasil tuning
best_params = {
    "colsample_bytree": 0.8,
    "learning_rate": 0.03,
    "max_depth": 9,
    "n_estimators": 500,
    "subsample": 0.8,
    "verbosity": 0,
    "random_state": 42,
}

# Buat pipeline
pipeline = Pipeline(
    steps=[
        ("encoder", TargetEncoder(cols=["location"])),
        ("scaler", MinMaxScaler()),
        ("regressor", XGBRegressor(**best_params)),
    ]
)

# Fit pipeline ke seluruh data
pipeline.fit(X, y)

# Simpan pipeline
joblib.dump(pipeline, "../../models/xgboost_pipeline.joblib")