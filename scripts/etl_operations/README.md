These scripts are used to benchmark ETL performance with Ray and Spark, assuming access to a shared HDFS at hdfs://okeanos-master:54310.

All generated datasets are stored directly in the /data directory within HDFS. The scripts assume this location for reading input data.

- Ray scripts require the HDFS path to the dataset.

- Spark scripts require both the number of executors and the HDFS path.

The datasets are expected to be a collection of CSV files with the schema produced by the data generation script included in this project.

Example:

Ray:

```bash
python3 <script_folder>/<script>.py hdfs:filepath
```
Spark:

```bash
spark-submit --packages "ch.cern.sparkmeasure:spark-measure_2.12:0.23" <script_folder/script> <num_executors> hdfs:filepath
```
