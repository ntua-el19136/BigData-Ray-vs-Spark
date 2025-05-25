## **Classification Data Generation**

This folder contains a script for generating large-scale classification datasets in CSV format. The dataset includes numerical features relevant to classification tasks, along with random categorical features. It is designed to benchmark ETL performance. The script writes directly to a configured HDFS instance, making it especially useful when generating datasets too large to fit on a single machine's local disk.

Use the script with:

```bash
./generate_data_to_hdfs.sh <num_samples> </path/in/hdfs.csv>
```
