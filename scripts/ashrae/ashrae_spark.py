# === Spark ML Pipeline for ASHRAE dataset ===
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, log1p
from sparkmeasure import StageMetrics
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import time, psutil
import sys

if len(sys.argv) < 2:
    print("Usage: spark-submit ashrae_pipeline_spark.py <num_executors>")
    sys.exit(1)

num_executors = sys.argv[1]

# === Init SparkSession ===
spark = SparkSession.builder \
    .appName("ASHRAE Spark ML Pipeline") \
    .master("yarn") \
    .config("spark.jars.packages", "ch.cern.sparkmeasure:spark-measure_2.12:0.23") \
    .config("spark.executor.instances", num_executors) \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .config("spark.driver.memory", "5g") \
    .config("spark.default.parallelism", "12") \
    .config("spark.executor.memoryOverhead", "1024") \
    .config("spark.sql.shuffle.partitions", "48") \
    .getOrCreate()

stagemetrics = StageMetrics(spark)
stagemetrics.begin()

# === Load Data ===
data_path = "hdfs://okeanos-master:54310/datasets/ashrae"
train_df = spark.read.csv(f"{data_path}/train.csv", header=True, inferSchema=True)
weather_df = spark.read.csv(f"{data_path}/weather_train.csv", header=True, inferSchema=True)
meta_df = spark.read.csv(f"{data_path}/building_metadata.csv", header=True, inferSchema=True)

# === Join datasets ===
train_df = train_df.join(meta_df, on="building_id", how="left")
train_df = train_df.join(weather_df, on=["site_id", "timestamp"], how="left")

# === Drop NA and apply log1p to target ===
train_df = train_df.dropna()
train_df = train_df.withColumn("log_meter_reading", log1p(col("meter_reading")))

# === Categorical Indexing & Encoding ===
indexer = StringIndexer(inputCol="primary_use", outputCol="primary_use_index")
ohe = OneHotEncoder(inputCols=["primary_use_index"], outputCols=["primary_use_ohe"])

# === Assemble features ===
features = [
    "square_feet", "air_temperature", "dew_temperature", "cloud_coverage", "precip_depth_1_hr",
    "primary_use_ohe"
]
assembler = VectorAssembler(inputCols=features, outputCol="features")

# === Standardization ===
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# === Model ===
gbt = GBTRegressor(labelCol="log_meter_reading", featuresCol="scaled_features")

# === Pipeline ===
preprocessing_stages = [indexer, ohe, assembler, scaler]

# === Param Grid ===
paramGrid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [5, 7]) \
    .addGrid(gbt.maxIter, [10, 20]) \
    .build()

# === Evaluator ===
evaluator = RegressionEvaluator(labelCol="log_meter_reading", predictionCol="prediction", metricName="rmse")

# === CrossValidator ===
cv = CrossValidator(
    estimator=Pipeline(stages=preprocessing_stages + [gbt]),
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3,
    parallelism=int(num_executors)
)

# === Train/Test Split ===
train_data, test_data = train_df.randomSplit([0.8, 0.2], seed=42)
train_data = train_data.cache()
train_data.count()

# === Timing ===
start = time.time()
model = cv.fit(train_data)
end = time.time()
train_duration = end - start

# === Evaluation ===
predictions = model.transform(test_data)
rmse = evaluator.evaluate(predictions)
r2 = RegressionEvaluator(
    labelCol="log_meter_reading", predictionCol="prediction", metricName="r2").evaluate(predictions)

# === Metrics ===
cpu = psutil.cpu_percent(interval=1)
ram = psutil.virtual_memory().used / (1024**3)

stagemetrics.end()

print("\n=== ASHRAE Spark ML Report ===")
print(f"Training duration : {train_duration:.2f} sec")
print(f"RMSE              : {rmse:.4f}")
print(f"R^2               : {r2:.4f}")
print(f"CPU usage         : {cpu} %")
print(f"RAM used          : {ram:.2f} GB")
print(f"Best Model Params : {model.bestModel.stages[-1].extractParamMap()}")
stagemetrics.print_report()
print(stagemetrics.aggregate_stagemetrics())

spark.stop()
