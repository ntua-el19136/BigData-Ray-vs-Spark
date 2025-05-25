from pyspark.sql import SparkSession
from sparkmeasure import StageMetrics
import sys
import os
import time

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

if len(sys.argv) != 3:
    print("Usage: sort_spark.py <num_executors> <input_path>")
    sys.exit(1)

num_executors = sys.argv[1]
input_path = sys.argv[2]

spark = SparkSession.builder \
    .appName("Spark Sort Job") \
    .master("yarn") \
    .config("spark.executor.instances", num_executors) \
    .config("spark.jars.packages", "ch.cern.sparkmeasure:spark-measure_2.12:0.23") \
    .getOrCreate()
sc = spark.sparkContext

stagemetrics = StageMetrics(spark)

start_time = time.time()
stagemetrics.begin()

df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)
sorted_df = df.sort("feature_2")
sorted_df.show(5)
count = sorted_df.count()

stagemetrics.end()
end_time = time.time()

print("🔢 Count after sort:", count)
stagemetrics.print_report()
metrics = stagemetrics.aggregate_stagemetrics()

print(f"⏱Total Runtime: {end_time - start_time:.2f} seconds")
print(f"Peak Heap Memory: {metrics.get('Peak JVM memory', 'N/A')} MB")
print(f"Total CPU Time: {metrics.get('executorCpuTime', 'N/A')} ms")
print(f"Total Tasks: {metrics.get('numTasks', 'N/A')}")

patience = 20
while patience > 0:
    try:
        stagemetrics.print_memory_report()
        break
    except:
        time.sleep(1)
        patience -= 1

spark.stop()
