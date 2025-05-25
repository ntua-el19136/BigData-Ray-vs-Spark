import ray
from pyarrow import fs
import sys
import time
import pandas as pd

ray.init()

start_time = time.time()

hdfs_fs = fs.HadoopFileSystem.from_uri("hdfs://okeanos-master:54310")

input_path = sys.argv[1]

ds = ray.data.read_csv(input_path, filesystem=hdfs_fs)
print("✅ Dataset loaded")

ds = ds.repartition(200)

def add_new_feature(batch: pd.DataFrame) -> pd.DataFrame:
    batch["new_feature"] = (batch["feature_1"] ** 2 + batch["feature_2"] ** 2).pow(0.5)
    return batch

ds = ds.map_batches(add_new_feature, batch_format="pandas", batch_size=64000)
print("➕ Υπολογίστηκε η στήλη 'new_feature'.")

def vectorized_filter(batch: pd.DataFrame) -> pd.DataFrame:
    mask = batch["word"].str.len() > batch["new_feature"]
    return batch[mask]

ds = ds.map_batches(vectorized_filter, batch_format="pandas", batch_size=64000)
print("🔍 Εφαρμόστηκε φίλτρο length(word) > new_feature.")

ds = ds.materialize()
print("📊 Δείγμα:")
ds.show(5)
print("📏 Σύνολο γραμμών:", ds.count())

end_time = time.time()
print(f"⏱️ Συνολικός χρόνος εκτέλεσης: {end_time - start_time:.2f} δευτερόλεπτα")

print(ds.stats())
