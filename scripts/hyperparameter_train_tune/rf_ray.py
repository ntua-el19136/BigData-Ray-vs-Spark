import ray
from ray.train.sklearn import SklearnTrainer
from ray.data.preprocessors import Chain, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from pyarrow import fs
from ray import tune
import sys
import time

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 rf_ray_air_tune.py <hdfs_path_to_csv> <num_workers>")
        sys.exit(1)

    hdfs_path = sys.argv[1]
    num_workers = int(sys.argv[2])

    ray.init()
    start = time.time()

    hdfs_fs = fs.HadoopFileSystem.from_uri("hdfs://okeanos-master:54310")
    ds = ray.data.read_csv(hdfs_path, filesystem=hdfs_fs)

    if "word" in ds.schema().names:
        ds = ds.drop_columns(["word"])

    def add_new_feature(batch):
        batch["new_feature"] = batch["feature_1"] ** 2 + batch["feature_2"] ** 2
        return batch

    ds = ds.map_batches(add_new_feature, batch_format="numpy")

    selected_columns = [
        "feature_1", "feature_2", "feature_3", "feature_4",
        "categorical_feature_1", "categorical_feature_2",
        "new_feature", "label"
    ]
    ds = ds.select_columns(selected_columns)

    preprocessing_time = time.time() - start

    feature_columns = [c for c in selected_columns if c != "label"]

    # Standard scaler
    scaler = StandardScaler(columns=[
        "feature_1", "feature_2", "feature_3", "feature_4", "new_feature"
    ])
    preprocessor = Chain([scaler])

    def trainer_config(config):
        return SklearnTrainer(
            estimator=RandomForestClassifier(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                max_features=config["max_features"],
                random_state=42,
                n_jobs=-1
            ),
            label_column="label",
            features=feature_columns,
            datasets={"train": ds},
            preprocessor=preprocessor,
            scaling_config={"num_workers": num_workers}
        )

    # Tuner
    tuner = tune.Tuner(
        trainable=trainer_config,
        param_space={
            "n_estimators": tune.choice([50, 100, 200]),
            "max_depth": tune.choice([5, 10, 20]),
            "max_features": tune.choice(["sqrt", "log2"])
        },
        tune_config=tune.TuneConfig(
            metric="train_accuracy",
            mode="max",
            num_samples=6
        ),
        run_config=tune.RunConfig()
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric="train_accuracy", mode="max")

    total_time = time.time() - start

    print("BEST RESULT CONFIG:", best_result.config)
    print("BEST RESULT METRICS:", best_result.metrics)
    print("Preprocessing time:", preprocessing_time)
    print("Total time:", total_time)
