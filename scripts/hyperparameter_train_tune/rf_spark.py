from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col, sqrt
import sys, os, time

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

num_workers = int(sys.argv[1])
relative_path = sys.argv[2]

# Initialize Spark session
spark = SparkSession.builder \
    .appName("rf_spark_crossval") \
    .master("yarn") \
    .config("spark.executor.instances", num_workers) \
    .config("spark.executor.cores", "4") \
    .getOrCreate()

start_time = time.time()

hdfs_base = "hdfs://okeanos-master:54310"
hdfs_path = f"{hdfs_base}{relative_path}"
df = spark.read.csv(hdfs_path, header=True, inferSchema=True)

if "word" in df.columns:
    df = df.drop("word")

df = df.withColumn("new_feature", sqrt(col("feature_1")**2 + col("feature_2")**2))

# Select only valid columns
selected_features = ["feature_1", "feature_2", "feature_3", "feature_4",
                     "categorical_feature_1", "categorical_feature_2", "new_feature", "label"]
df = df.select(*[c for c in selected_features if c in df.columns])

# Assemble + Scale
assembler = VectorAssembler(
    inputCols=[c for c in df.columns if c != "label"],
    outputCol="input"
)
scaler = MinMaxScaler(inputCol="input", outputCol="features")
pipeline = Pipeline(stages=[assembler, scaler])
processed_df = pipeline.fit(df).transform(df).select("features", "label")

# Split dataset
train, test = processed_df.randomSplit([0.7, 0.3], seed=42)

# Define classifier and evaluator
rf = RandomForestClassifier(labelCol="label", featuresCol="features")
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)

# Define parameter grid
param_grid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [5]) \
    .addGrid(rf.maxDepth, [10) \
    .addGrid(rf.featureSubsetStrategy, ["sqrt", "log2"]) \
    .build()

# CrossValidator
crossval = CrossValidator(
    estimator=rf,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=1,
    parallelism=num_workers
)

# Fit and evaluate
cv_model = crossval.fit(train)
predictions = cv_model.transform(test)
accuracy = evaluator.evaluate(predictions)

print("✅ BEST MODEL ACCURACY:", accuracy)
print("✅ BEST PARAMETERS:", cv_model.bestModel.extractParamMap())
print("⏱ Total time:", time.time() - start_time)

spark.stop()
