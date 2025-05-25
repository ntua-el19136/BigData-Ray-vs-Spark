## **Classification Data Generation**

This folder contains a script for generating large-scale classification datasets in CSV format. The dataset includes numerical features relevant to classification tasks, along with random categorical features. It is designed to benchmark ETL performance. The script writes directly to a configured HDFS instance, making it especially useful when generating datasets too large to fit on a single machine's local disk.

Use the script with:

```bash
./upload_data_to_hdfs.sh <num_samples> <hdfs_path>
```
## **Graph Data Generation**

This script generates synthetic graph datasets with customizable parameters and uploads them directly to a configured HDFS instance. Users can specify the number of nodes and select the graph type — including random, scale-free, or small-world structures. This tool is designed to support performance benchmarking of distributed graph processing operations such as PageRank and triangle counting, and is particularly useful for generating graphs too large to store on a single machine.

Use the script with:

```bash
./upload_graph_to_hdfs.sh <num_nodes> <graph_type> <hdfs_path>
```
