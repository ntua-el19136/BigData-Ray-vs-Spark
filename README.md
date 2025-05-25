# **NTUA-BigData-Ray-vs-Spark**

### **Comparison of Python's Scaling Frameworks: Ray vs. Apache Spark**

**Analysis and Design of Information Systems – Team 27**

## Researchers
- Athanasios Bitsanis – 03119136 – ntua-el19136

- Stefanos Theos – 03119219 – ntua-el19219

## **Project Overview**

This repository includes all necessary data generation tools and execution scripts used in a comparative study between Ray and Apache Spark — two prominent distributed computing frameworks. The study was conducted as part of the Analysis and Design of Information Systems course at NTUA.

## **Research Objective**

The goal of this project is to evaluate and compare the performance and scalability of Ray and Apache Spark across several categories of large-scale distributed data processing tasks. The experiments were designed to assess runtime performance, resource efficiency, and framework flexibility under various workloads and cluster configurations.

All tasks were executed on datasets ranging from 2 GB to 8 GB in size — beyond the capacity of a single machine’s memory — using a 3-node cluster (each VM equipped with 4 CPUs and 8 GB RAM).

## **Evaluated Tasks**

**ETL Operations on Big Datasets (CSVs)**
- Benchmarked aggregate, transform, load, sort operations on large CSV files.
- Focused on processing speed, cluster resource utilization, and scalability.

**Graph Operations (PageRank, Triangle Counting)**
- Implemented PageRank to identify top-ranking nodes across various graph sizes.
- Performed triangle counting to evaluate performance on local graph computations.

**ML Operations and Pipelines**
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
   
**Energy consumption prediction using the ASHRAE dataset (distributed preprocessing, training, and hyperparameter tuning on tabular sensor data)**

These experiments aimed to evaluate runtime efficiency, tuning flexibility, and final model performance (accuracy and AUC) across different data sizes and cluster configurations.
