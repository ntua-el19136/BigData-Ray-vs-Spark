import ray
import pandas as pd
import numpy as np
import subprocess
import argparse
import time
import psutil

parser = argparse.ArgumentParser()
parser.add_argument("hdfs_path", help="Path to .tsv graph file on HDFS")
args = parser.parse_args()

start_time = time.time()

cat = subprocess.Popen(["hadoop", "fs", "-cat", args.hdfs_path], stdout=subprocess.PIPE)
lines = [
    tuple(line.decode().strip().split('\t'))
    for line in cat.stdout
    if not line.startswith(b'#') and '\t' in line.decode()
]
edges_list = [{'FromNode': t[0], 'ToNode': t[1]} for t in lines]

if not edges_list:
    print("❌ No edges found. Exiting.")
    exit(1)

print(f"✅ Loaded {len(edges_list)} edges from {args.hdfs_path}")
print(f"▶ Sample edge: {edges_list[0]}")

ray.init(ignore_reinit_error=True)

edges_ds = ray.data.from_items(edges_list).repartition(8)

def rename_column(batch: pd.DataFrame, old: str, new: str) -> pd.DataFrame:
    return batch.rename(columns={old: new})

from_nodes = edges_ds.select_columns(["FromNode"]).map_batches(
    lambda df: rename_column(df, "FromNode", "Node"),
    batch_format="pandas"
)

to_nodes = edges_ds.select_columns(["ToNode"]).map_batches(
    lambda df: rename_column(df, "ToNode", "Node"),
    batch_format="pandas"
)

unioned = from_nodes.union(to_nodes)
all_nodes = unioned.groupby("Node").count().drop_columns(["count()"])

N = all_nodes.count()
print(f"📦 Total nodes: {N}")

# Initialize PageRank
nodes = all_nodes.add_column("PageRank", lambda df: np.ones(len(df)))

outgoing = edges_ds.groupby("FromNode").count() \
    .rename_columns({"FromNode": "Node", "count()": "NumOut"})

nodes_df = nodes.to_pandas()
outgoing_df = outgoing.to_pandas()

# PageRank parameters
alpha = 0.85
num_iters = 10

# PageRank Iterations
for i in range(num_iters):
    print(f"🔁 Iteration {i+1}/{num_iters}")

    def join_with_node_info(batch: pd.DataFrame) -> pd.DataFrame:
        batch = batch.merge(nodes_df, how="left", left_on="FromNode", right_on="Node").drop(columns=["Node"])
        batch = batch.merge(outgoing_df, how="left", left_on="FromNode", right_on="Node").drop(columns=["Node"])
        return batch

    edges_with_info = edges_ds.map_batches(
        join_with_node_info,
        batch_format="pandas"
    )

    # Calculate contributions
    contribs = edges_with_info.map_batches(
        lambda df: pd.DataFrame({
            "Node": df["ToNode"],
            "contrib": df["PageRank"] / df["NumOut"]
        }),
        batch_format="pandas"
    ).groupby("Node").sum("contrib")

    # Update PageRank
    nodes = contribs.map_batches(
        lambda df: df.assign(PageRank=0.15 + alpha * df["sum(contrib)"]).drop(columns=["sum(contrib)"]),
        batch_format="pandas"
    )

    nodes_df = nodes.to_pandas()

nodes = nodes.sort("PageRank", descending=True)
print("\n📈 Top 10 nodes by PageRank:")
nodes.show(10)

end_time = time.time()
print(f"\n⏱ Total execution time: {end_time - start_time:.2f} seconds")
cpu_percent = psutil.cpu_percent(interval=1)
print(f"🧠 CPU usage (local): {cpu_percent}%")
