import ray
from pyarrow import fs
import pandas as pd
import sys

ray.init(address="auto")

# HDFS filesystem
hdfs_fs = fs.HadoopFileSystem.from_uri("hdfs://okeanos-master:54310")

input_path = sys.argv[1]

ds = ray.data.read_csv(input_path, filesystem=hdfs_fs)

total_rows = ds.count()
half = total_rows // 2

ds1, ds2 = ds.split_at_indices([half])

right_pd = ds2.to_pandas()

join_key = "word"

def join_func(batch: pd.DataFrame) -> pd.DataFrame:
    return batch.merge(right_pd, on=join_key, how="inner")

joined = ds1.map_batches(join_func, batch_format="pandas")

print("Join ολοκληρώθηκε. Δείγμα:")
print(joined.show(5))

print("Στατιστικά joined dataset:")
joined = joined.materialize()
print(joined.stats())
