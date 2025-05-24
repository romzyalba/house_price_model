import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

# mlflow track
mlflow.set_tracking_uri("http://localhost:5000")

# Load data
rumah123_df = pd.read_csv("data/rumah123_ready.csv")
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

# Mulai MLflow run
with mlflow.start_run(run_name="XGBoost_House_Price_Pipeline"):

    # Fit pipeline ke seluruh data
    pipeline.fit(X, y)
    
    y_pred = pipeline.predict(X)
    
    # Hitung metrik evaluasi
    rmse = mean_squared_error(y, y_pred, squared=False)
    
    # Log parameter
    mlflow.log_params(best_params)
    
    # Log metrik
    mlflow.log_metric("rmse", rmse)

    
    # Log model pipeline
    mlflow.sklearn.log_model(pipeline, artifact_path="model")
    
    # Simpan pipeline (tak comment ben ndak di run berulang ulang)
    # joblib.dump(pipeline, "../../models/xgboost_pipeline.joblib")
    
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"RMSE: {rmse:.4f}")
