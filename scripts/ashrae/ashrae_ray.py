import ray
from ray import train, tune
from ray.data import read_csv
from ray.train.xgboost import XGBoostTrainer
from ray.train import ScalingConfig
import time
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import psutil
import os
from ray.air import session

# =========================
# Load Data from HDFS
# =========================
print("Loading datasets from HDFS...")
buildings = read_csv("hdfs://okeanos-master:54310/datasets/ashrae/building_metadata.csv")
weather_train = read_csv("hdfs://okeanos-master:54310/datasets/ashrae/weather_train.csv")
train_data = read_csv("hdfs://okeanos-master:54310/datasets/ashrae/train.csv")

# =========================
# Sample data to reduce volume
# =========================
print("Sampling data...")
train_data = train_data.random_sample(0.005)

# =========================
# Manual Join using map_batches
# =========================
print("Joining datasets manually with map_batches...")

# Convert buildings and weather to pandas DataFrames
buildings_df = buildings.to_pandas()
weather_train_df = weather_train.to_pandas()

# Join with buildings metadata
train_data = train_data.map_batches(
    lambda df: df.merge(buildings_df, on="building_id", how="left"),
    batch_format="pandas",
    batch_size=1000
)

# Join with weather data
train_data = train_data.map_batches(
    lambda df: df.merge(weather_train_df, on=["site_id", "timestamp"], how="left"),
    batch_format="pandas",
    batch_size=1000
)

# =========================
# Preprocessing
# =========================
print("Preprocessing...")
columns_to_drop = ["timestamp", "row_id"]
existing_cols = [col for col in columns_to_drop if col in train_data.schema().names]
if existing_cols:
    train_data = train_data.drop_columns(existing_cols)

train_data = train_data.filter(lambda row: all(v is not None for v in row.values()))

if "primary_use" in train_data.schema().names:
    train_data = train_data.map_batches(
        lambda df: df.assign(primary_use=df["primary_use"].astype("category").cat.codes),
        batch_format="pandas",
        batch_size=1000
    )

train_data = train_data.materialize()

# =========================
# Train/Test Split
# =========================
print("Splitting dataset...")
all_data = train_data.random_shuffle()
row_count = all_data.count()
train_size = int(row_count * 0.8)
train_ds, test_ds = all_data.split_at_indices([train_size])

label_column = "meter_reading"
feature_columns = [col for col in train_data.schema().names if col != label_column]

def trainable(config):
    trainer = XGBoostTrainer(
        run_config=train.RunConfig(storage_path="/mnt/shared"),
        label_column=label_column,
        params={
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "eta": config["eta"],
            "max_depth": config["max_depth"],
            "early_stopping_rounds": 10
        },
        scaling_config=ScalingConfig(
            num_workers=3,
            use_gpu=False,
            resources_per_worker={"CPU": 2}
        ),
        datasets={
            "train": train_ds,
            "validation": test_ds
        },
        num_boost_round=50
    )
    result = trainer.fit()
    #metrics = result.metrics
    #tune.report(rmse=metrics["validation"]["rmse"][-1])
    metrics = result.metrics
    print("Returned metrics:", metrics)

    rmse = metrics.get("ray_air:validation/rmse") or metrics.get("validation-rmse")
    if rmse is None:
        raise ValueError("Validation RMSE not found in result metrics!")

    session.report({
    "rmse": float(metrics.get("validation-rmse", np.nan)),
    "train_rmse": float(metrics.get("train-rmse", np.nan))
    })

# =========================
# Tuning
# =========================
print("Starting hyperparameter tuning...")
start_time = time.time()
analysis = tune.Tuner(
    trainable,
    param_space={
        "eta": tune.grid_search([0.1, 0.3]),
        "max_depth": tune.grid_search([6, 8])
    },
    tune_config=tune.TuneConfig(
        num_samples=1,
        max_concurrent_trials=1
    )
).fit()
end_time = time.time()

best_result = analysis.get_best_result(metric="rmse", mode="min")
print("Best hyperparameters:", best_result.config)

# =========================
# Report
# =========================
print("\n=== ASHRAE Ray Tune Report ===")
print(f"Tuning duration   : {end_time - start_time:.2f} sec")
print(f"Best RMSE         : {best_result.metrics['rmse']:.4f}")
print(f"Best config       : {best_result.config}")

cpu_percent = psutil.cpu_percent(interval=1)
virtual_mem = psutil.virtual_memory()
print(f"CPU usage         : {cpu_percent:.1f} %")
print(f"RAM used          : {virtual_mem.used / (1024**3):.2f} GB")
