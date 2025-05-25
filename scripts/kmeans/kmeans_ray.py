import ray
import argparse
import numpy as np
import pandas as pd
from ray.data.preprocessors import Concatenator
from sklearn.cluster import KMeans
import psutil
import time
from pyarrow import fs

parser = argparse.ArgumentParser()
parser.add_argument("hdfs_path", help="Path to CSV in HDFS")
parser.add_argument("--k", type=int, default=2, help="Number of clusters")
args = parser.parse_args()

print(f"🔄 Reading file from HDFS: {args.hdfs_path}")

# --- Init Ray ---
ray.init(address="auto")

hdfs = fs.HadoopFileSystem.from_uri("hdfs://okeanos-master:54310")
ds = ray.data.read_csv(args.hdfs_path, filesystem=hdfs)

df = ds.to_pandas()
features = df.select_dtypes(include=[np.number])
if features.shape[1] == 0:
    print("❌ No numeric columns to cluster on.")
    exit(1)

ds = ray.data.from_pandas(features)

feature_columns = list(features.columns)
preprocessor = Concatenator(columns=feature_columns, output_column_name="features")
ds = preprocessor.fit_transform(ds)

start_time = time.time()

X = np.array([row["features"] for row in ds.iter_rows()])

# --- KMeans clustering ---
kmeans = KMeans(n_clusters=args.k, random_state=42, n_init="auto")
clusters = kmeans.fit_predict(X)

end_time = time.time()
elapsed = end_time - start_time

# --- Cluster stats ---
from collections import Counter
counts = Counter(clusters)

print("\n✅ KMeans Clustering Report")
print(f"Total rows         : {X.shape[0]}")
print(f"Number of features : {X.shape[1]}")
print(f"Clusters (k)       : {args.k}")
print(f"Time taken         : {elapsed:.2f} sec")
print(f"CPU usage          : {psutil.cpu_percent(interval=1)}%")
print(f"RAM used           : {psutil.virtual_memory().used / (1024**3):.2f} GB")
print("Cluster sizes:")
for cid, count in sorted(counts.items()):
    print(f"  Cluster {cid}: {count} points")

ray.shutdown()
