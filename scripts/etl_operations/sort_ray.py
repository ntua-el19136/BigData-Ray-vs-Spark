import ray
from pyarrow import fs
import sys
import time
import psutil
import os

if len(sys.argv) != 3:
    print("Usage: sort_ray.py <num_workers> <input_path>")
    sys.exit(1)

num_workers = int(sys.argv[1])
input_path = sys.argv[2]

ray.init()

ctx = ray.data.DataContext.get_current()
ctx.use_push_based_shuffle = True
ctx.execution_options.resource_limits.cpu = num_workers

hdfs_fs = fs.HadoopFileSystem.from_uri("hdfs://okeanos-master:54310")

# Start metrics
start_time = time.time()
process = psutil.Process(os.getpid())
cpu_start = psutil.cpu_times_percent()
mem_start = process.memory_info().rss

# Read CSV and repartition based on desired parallelism
ds = ray.data.read_csv(input_path, filesystem=hdfs_fs)
ds = ds.map_batches(lambda batch: batch)

ds_sorted = ds.sort("feature_2")

ds_sorted.show(limit=5)

end_time = time.time()
cpu_end = psutil.cpu_times_percent()
mem_end = process.memory_info().rss

print("Stats:", ds.materialize().stats())
print(f"⏱Total Runtime: {end_time - start_time:.2f} seconds")
print(f"Peak Heap Memory (Process RSS): {(mem_end / 1024**2):.2f} MB")
print(f"CPU Times % (user/system): {cpu_end.user:.1f}% / {cpu_end.system:.1f}%")
