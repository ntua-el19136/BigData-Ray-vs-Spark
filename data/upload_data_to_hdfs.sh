#!/bin/bash

# This script runs generate_dataset.py to create classification data
# and pipes it directly into HDFS at a specified path.

NUM_SAMPLES=${1:-5000}
HDFS_DATA_PATH=${2:-/data/generated_data.csv}

# Run Python generator and pipe output to HDFS
python3 data_generator.py "$NUM_SAMPLES" | hdfs dfs -put -f - "$HDFS_DATA_PATH"
