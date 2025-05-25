# **NTUA-BigData-Ray-vs-Spark**

### **Comparison of Python's Scaling Frameworks: Ray vs. Apache Spark**

**Analysis and Design of Information Systems – Team 27**

## Researchers
- Athanasios Bitsanis – 03119136

- Stefanos Theos – 03119219

## **Project Overview**

This repository includes all necessary data generation tools and execution scripts used in a comparative study between Ray and Apache Spark — two prominent distributed computing frameworks. The study was conducted as part of the Analysis and Design of Information Systems course at NTUA.

## **Research Objective**

The goal of this project is to evaluate and compare the performance and scalability of Ray and Apache Spark across several categories of large-scale distributed data processing tasks. The experiments were designed to assess runtime performance, resource efficiency, and framework flexibility under various workloads and cluster configurations.

All tasks were executed on datasets ranging from 2 GB to 8 GB in size — beyond the capacity of a single machine’s memory — using a 3-node cluster (each VM equipped with 4 CPUs and 8 GB RAM).

## **Evaluation Tasks**

**ETL Operations on Big Datasets (CSVs):**
- Benchmarked aggregate, transform, load, sort operations on large CSV files.
- Focused on processing speed, cluster resource utilization, and scalability.

**Graph Operations (PageRank, Triangle Counting):**
- Implemented PageRank to identify top-ranking nodes across various graph sizes.
- Performed triangle counting to evaluate performance on local graph computations.

**ML Operations and Pipelines:**
- **K-Means Clustering**
  - Applied K-means on large datasets to compare clustering performance and runtime across frameworks.

- **Hyperparameter Training and Tuning**
Implemented complete training and hyperparameter tuning workflows for two different models:

  - #### Neural Networks using:

    - Ray with PyTorch and Ray Tune

    - Spark with MLlib's pipeline-based tuning

  - #### Random Forest Classifiers using:

    - Ray with Scikit-learn (via Ray Train/Tune)
    - Spark with MLlib
   
**Energy consumption prediction using the ASHRAE dataset:**

- Distributed preprocessing, training, and hyperparameter tuning on tabular sensor data

## **Requirements**
Each VM should have a Python virtual environment set up. Install dependencies with:

```bash
pip install -r requirements.txt
```


To install additional Ray components:

```bash
pip install ray[core,data,train,tune]
```

## **Installation**
To configure HDFS, YARN, and Spark, we followed the setup guide from the NTUA Advanced Databases course:

[Hadoop  and Spark Setup Guide (Colab)](https://colab.research.google.com/drive/1pjf3Q6T-Ak2gXzbgoPpvMdfOHd1GqHZG?usp=sharing#scrollTo=AVipleZma-DY)

Ray setup only requires Python packages and does not depend on external services.

##  **Cluster Setup**
**Spark (Head Node):**

```bash
start-dfs.sh
start-yarn.sh
$SPARK_HOME/sbin/start-history-server.sh
```

**Ray (Head Node):**

```bash
ray start --head --node-ip-address=[head-node-private-ip-address] --port=6379 --dashboard-host=0.0.0.0 --object-store-memory=2147483648 --system-config='{"automatic_object_spilling_enabled": true, "object_spilling_threshold": 0.8}'
```

**Ray (Worker Nodes):**

```bash
ray start --address=[head-node-private-ip-address]
```

## **How to Run Experiments**
**Spark Scripts**

```bash
spark-submit --packages "ch.cern.sparkmeasure:spark-measure_2.12:0.23" <script_folder>/<script> <num_executors> <hdfs:filepath>
```

**Ray Scripts**

```bash
python3 <script_folder>/<script>.py <hdfs:filepath>
```
