import ray
import networkx as nx
import subprocess
import time
import sys

if len(sys.argv) != 3:
    print("Usage: python triangle_networkx_ray.py <num_chunks> <hdfs_path>")
    exit(1)

num_chunks = int(sys.argv[1])
hdfs_path = sys.argv[2]

print(f"🔄 Reading graph from: {hdfs_path}")
cat = subprocess.Popen(["hadoop", "fs", "-cat", hdfs_path], stdout=subprocess.PIPE)
lines = [line.decode().strip() for line in cat.stdout if not line.startswith(b"#") and '\t' in line.decode()]

if not lines:
    print("❌ No valid edges found in the file.")
    exit(1)

# --- Parse to NetworkX graph ---
G = nx.parse_edgelist(lines, delimiter='\t', nodetype=int, create_using=nx.DiGraph()).to_undirected()

# --- Split nodes into chunks ---
node_list = list(G.nodes())
node_chunk_size = max(1, len(node_list) // num_chunks)

node_chunks = [
    node_list[i * node_chunk_size : (i + 1) * node_chunk_size] if i < num_chunks - 1
    else node_list[i * node_chunk_size :]
    for i in range(num_chunks)
]

# --- Ray init ---
ray.init(ignore_reinit_error=True)

@ray.remote
def compute_triangles(G, nodes):
    return nx.triangles(G, nodes=nodes)

start = time.time()

G_ref = ray.put(G)
futures = [compute_triangles.remote(G_ref, chunk) for chunk in node_chunks]
results = ray.get(futures)

# --- Combine results ---
triangle_counts = sum(sum(d.values()) for d in results) // 3

duration = time.time() - start

print("\n=== Triangle Counting Report (Ray + NetworkX) ===")
print(f"Total triangles    : {triangle_counts}")
print(f"Total nodes        : {len(G.nodes())}")
print(f"Total edges        : {len(G.edges())}")
print(f"Chunks (Ray tasks) : {num_chunks}")
print(f"Elapsed time       : {duration:.2f} sec")
print("=============================================")

