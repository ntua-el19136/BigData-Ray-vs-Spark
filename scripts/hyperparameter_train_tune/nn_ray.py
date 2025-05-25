import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from ray import tune, air

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

import random
import os
import sys
import time

# ------------------------ PyTorch Dataset ------------------------
class TabularDataset(Dataset):
    def __init__(self, data):
        self.X = torch.tensor([x["concat_out"] for x in data], dtype=torch.float32)
        self.y = torch.tensor([x["label"] for x in data], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ------------------------ PyTorch MLP ------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.out(x))
        return x

# ------------------------ Training Function ------------------------
def train_mlp(config, train_ds=None, val_ds=None):
    train_iter = train_ds.iter_torch_batches(batch_size=64)
    val_iter = val_ds.iter_torch_batches(batch_size=256)

    model = SimpleMLP(input_dim=7)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.BCELoss()

    for epoch in range(config["epochs"]):
        model.train()
        for batch in train_iter:
            X = batch["concat_out"]
            y = batch["label"]
            optimizer.zero_grad()
            y_pred = model(X).squeeze()
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    y_preds, y_trues = [], []
    with torch.no_grad():
        for batch in val_iter:
            X = batch["concat_out"]
            y = batch["label"]
            pred = model(X).squeeze()
            y_preds.extend(pred.numpy())
            y_trues.extend(y.numpy())

    auc = roc_auc_score(y_trues, y_preds)
    tune.report(auc=auc)

# ------------------------ Main ------------------------
if __name__ == "__main__":
    start_time = time.time()

    if len(sys.argv) < 2:
        print("Usage: python hyperparameter_ray.py <relative_hdfs_path>")
        sys.exit(1)

    ray.init()

    relative_path = sys.argv[1]
    hdfs_base = "hdfs://okeanos-master:54310"
    hdfs_path = f"{hdfs_base}{relative_path}"

    print(f"🚀 Reading from: {hdfs_path}")
    ds = ray.data.read_csv(hdfs_path)

    if "word" in ds.schema().names:
        ds = ds.drop_columns(["word"])

    ds = ds.map(lambda row: {
        **row,
        "new_feature": row["feature_1"]**2 + row["feature_2"]**2
    })

    feature_keys = [
        "feature_1", "feature_2", "feature_3", "feature_4",
        "categorical_feature_1", "categorical_feature_2",
        "new_feature"
    ]

    ds = ds.map(lambda row: {
        "concat_out": [row[k] for k in feature_keys],
        "label": float(row["label"])
    })

    def is_valid(row):
        return isinstance(row["concat_out"], list) and len(row["concat_out"]) == 7

    ds = ds.filter(is_valid)

    ds = ds.map(lambda row: {**row, "_rand": random.random()})
    train_ds = ds.filter(lambda row: row["_rand"] < 0.7).drop_columns(["_rand"])
    val_ds = ds.filter(lambda row: row["_rand"] >= 0.7).drop_columns(["_rand"])

    param_space = {
        "lr": tune.grid_search([0.01]),
        "epochs": tune.grid_search([1]),
    }

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_mlp, train_ds=train_ds, val_ds=val_ds),
            resources={"cpu": 2}
        ),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            metric="auc",
            mode="max",
            num_samples=1
        ),
        run_config=air.RunConfig(
            name="mlp_tuning"
        )
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric="auc", mode="max")

    print("\nBest Config:")
    print(best_result.config)
    print(f"Best AUC: {best_result.metrics['auc']:.4f}")
    print(f"Total runtime: {time.time() - start_time:.2f} seconds")

    ray.shutdown()
