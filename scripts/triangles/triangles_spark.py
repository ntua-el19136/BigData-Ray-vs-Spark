from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from sparkmeasure import StageMetrics
from pyspark.sql.functions import col, sum as spark_sum
from pyspark.sql.types import StructType, StructField, StringType
from graphframes import *
import os
import sys
import time
import psutil

# === Init ===
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

sparky = SparkSession \
    .builder \
    .appName("triangle counting") \
    .master("yarn") \
    .config("spark.executor.instances", sys.argv[1]) \
    .config("spark.jars.packages", "ch.cern.sparkmeasure:spark-measure_2.12:0.23,graphframes:graphframes:0.8.3-spark3.5-s_2.12") \
    .getOrCreate()

sc = sparky.sparkContext
stagemetrics = StageMetrics(sparky)

# === Schema & Data ===
schema = StructType([
    StructField("src", StringType(), True),
    StructField("dst", StringType(), True)
])

edges_df = sparky.read.format("csv") \
    .option("header", "false") \
    .option("delimiter", "\t") \
    .schema(schema) \
    .load("hdfs://okeanos-master:54310" + sys.argv[2])

vertices_df = edges_df \
    .select("src").union(edges_df.select("dst")) \
    .distinct() \
    .withColumnRenamed('src', 'id')

graph = GraphFrame(vertices_df, edges_df)

# === Triangle Counting ===
stagemetrics.begin()
start_time = time.time()

triangle_df = graph.triangleCount()
total_triangles = triangle_df.select(spark_sum("count")).collect()[0][0] // 3

end_time = time.time()
elapsed = end_time - start_time
stagemetrics.end()

# === Collect Metrics ===
n_nodes = vertices_df.count()
n_edges = edges_df.count()
cpu = psutil.cpu_percent(interval=1)
ram = psutil.virtual_memory().used / (1024 ** 3)

print("\n=== Triangle Counting Report (Spark) ===")
print(f"Total triangles: {total_triangles}")
print(f"Unique nodes   : {n_nodes}")
print(f"Total edges    : {n_edges}")
print(f"Executors      : {sys.argv[1]}")
print(f"Elapsed time   : {elapsed:.2f} sec")
print(f"CPU usage      : {cpu}%")
print(f"RAM used       : {ram:.2f} GB")
print("========================================")

# === SparkMeasure Reports ===
stagemetrics.print_report()
print(stagemetrics.aggregate_stagemetrics())

patience = 20
while patience > 0:
    try:
        stagemetrics.print_memory_report()
        print("memory report printed")
        break
    except:
        print("memory report not ready")
        time.sleep(1)
        patience -= 1
if patience == 0:
    print("memory report never ready :(")

sc.stop()

