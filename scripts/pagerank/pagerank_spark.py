from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sparkmeasure import StageMetrics
from graphframes import GraphFrame
import sys
import os
import time

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

if len(sys.argv) != 3:
    print("Usage: pagerank_spark.py <num_executors> <hdfs_graph_path>")
    sys.exit(1)

num_executors = sys.argv[1]
graph_path = sys.argv[2]

# Init SparkSession with GraphFrames and SparkMeasure
spark = SparkSession.builder \
    .appName("Spark PageRank with Metrics") \
    .master("yarn") \
    .config("spark.executor.instances", num_executors) \
    .config("spark.jars.packages",
            "graphframes:graphframes:0.8.2-spark3.1-s_2.12,ch.cern.sparkmeasure:spark-measure_2.12:0.23") \
    .getOrCreate()

sc = spark.sparkContext

stagemetrics = StageMetrics(spark)
stagemetrics.begin()

edges = spark.read.csv(graph_path, sep="\t", inferSchema=True).toDF("src", "dst")
vertices = edges.select(col("src").alias("id")).union(edges.select(col("dst").alias("id"))).distinct()

# Build graph and compute PageRank
g = GraphFrame(vertices, edges)
results = g.pageRank(resetProbability=0.15, maxIter=10)

# Show top 10 nodes by PageRank
results.vertices.select("id", "pagerank").orderBy(col("pagerank").desc()).show(10)

stagemetrics.end()

stagemetrics.print_report()
print(stagemetrics.aggregate_stagemetrics())

patience = 20
while patience > 0:
    try:
        stagemetrics.print_memory_report()
        print("Memory report printed.")
        break
    except:
        print("Waiting for memory report...")
        time.sleep(1)
        patience -= 1
else:
    print("Memory report never ready.")

sc.stop()
