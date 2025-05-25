from pyspark.sql import SparkSession
from sparkmeasure import StageMetrics
import sys
import time
import os

# Set Python executable
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Args: num_executors input_path
if len(sys.argv) != 3:
    print("Usage: join_spark.py <num_executors> <input_path>")
    sys.exit(1)

num_executors = sys.argv[1]
input_path = sys.argv[2]

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Spark Join Broadcast (Split Dataset)") \
    .master("yarn") \
    .config("spark.executor.instances", num_executors) \
    .config("spark.jars.packages", "ch.cern.sparkmeasure:spark-measure_2.12:0.23") \
    .getOrCreate()

sc = spark.sparkContext

# Read CSV
df = spark.read.option("header", "true").csv("hdfs://okeanos-master:54310/"+input_path, inferSchema=True)

# Count total rows and calculate midpoint
total_rows = df.count()
half = total_rows // 2

# Split dataset
df1 = df.limit(half)
df2 = df.subtract(df1)

# Find common join key
common_keys = set(df1.columns).intersection(df2.columns)
if not common_keys:
    print("No common join key found.")
    sys.exit(1)

join_key = list(common_keys)[0]
print(f"Performing join on column: '{join_key}'")

# Metrics start
stagemetrics = StageMetrics(spark)
stagemetrics.begin()

# Broadcast join
from pyspark.sql.functions import broadcast
joined_df = df1.join(broadcast(df2), on=join_key, how="inner")

# Trigger computation
joined_df.show(5)
count = joined_df.count()
print(f"Join completed. Rows after join: {count}")

# End metrics
stagemetrics.end()
stagemetrics.print_report()

# Aggregate stage metrics
print(stagemetrics.aggregate_stagemetrics())
print("join")

# Memory report (if available)
patience = 20
while patience > 0:
    try:
        stagemetrics.print_memory_report()
        print("print_memory_report")
        break
    except:
        print("memory report not ready")
        time.sleep(1)
        patience -= 1
else:
    print("memory report never ready :(")

# Cleanup
sc.stop()
