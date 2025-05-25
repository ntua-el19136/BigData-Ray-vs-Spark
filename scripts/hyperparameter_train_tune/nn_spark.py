from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pow
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

import time
import sys

# ------------------------ Main ------------------------
if __name__ == "__main__":
    start_time = time.time()

    if len(sys.argv) < 3:
        print("Usage: spark-submit nn_spark.py <relative_hdfs_path> <num_executors>")
        sys.exit(1)

    hdfs_base = "hdfs://okeanos-master:54310"
    relative_path = sys.argv[1]
    hdfs_path = f"{hdfs_base}{relative_path}"

    num_executors = int(sys.argv[2])

    spark = SparkSession.builder \
        .appName("Spark NN Hyperparameter Tuning") \
        .config("spark.executor.instances", num_executors) \
        .getOrCreate()

    print(f"🚀 Reading from: {hdfs_path}")
    print(f"🔧 Using {num_executors} executor(s)")

    # ------------------------ Load Data ------------------------
    df = spark.read.csv(hdfs_path, header=True, inferSchema=True)

    if "word" in df.columns:
        df = df.drop("word")

    df = df.withColumn("new_feature", pow(col("feature_1"), 2) + pow(col("feature_2"), 2))

    selected_cols = [
        "feature_1", "feature_2", "feature_3", "feature_4",
        "categorical_feature_1", "categorical_feature_2",
        "new_feature", "label"
    ]
    df = df.select(selected_cols)

    # ------------------------ Feature Engineering ------------------------
    feature_cols = selected_cols[:-1]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_assembled")
    scaler = MinMaxScaler(inputCol="features_assembled", outputCol="features")

    # ------------------------ Classifier ------------------------
    mlp = MultilayerPerceptronClassifier(
        labelCol="label", featuresCol="features", predictionCol="prediction",
        seed=1234, maxIter=10
    )

    pipeline = Pipeline(stages=[assembler, scaler, mlp])

    train_df, val_df = df.randomSplit([0.7, 0.3], seed=42)

    param_grid = ParamGridBuilder() \
        .addGrid(mlp.layers,
            [[7, 128, 2]]) \
        .addGrid(mlp.stepSize, [0.01]) \
        .addGrid(mlp.maxIter, [10]) \
        .build()

    evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=2,
        parallelism=num_executors
    )

    cv_model = cv.fit(train_df)

    best_model = cv_model.bestModel
    predictions = best_model.transform(val_df)
    auc = evaluator.evaluate(predictions)

    print("\n✅ Best Model Parameters:")
    print(best_model.stages[-1].extractParamMap())
    print(f"📈 Validation AUC: {auc:.4f}")
    print(f"⏱️ Total execution time: {time.time() - start_time:.2f} seconds")

    spark.stop()
