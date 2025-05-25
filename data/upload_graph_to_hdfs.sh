#!/bin/bash

NUM_NODES=${1:-10000}
GRAPH_TYPE=${2:-smallworld}
HDFS_PATH=${3:-/graphs}

INDEX=1
BASE_NAME="graph${GRAPH_TYPE}${NUM_NODES}.tsv"
FILENAME="$BASE_NAME"
HDFS_FULL_PATH="${HDFS_PATH}/${FILENAME}"

while hdfs dfs -test -e "$HDFS_FULL_PATH"; do
  INDEX=$((INDEX + 1))
  FILENAME="graph${INDEX}${GRAPH_TYPE}${NUM_NODES}.tsv"
  HDFS_FULL_PATH="${HDFS_PATH}/${FILENAME}"
done

hdfs dfs -test -d "$HDFS_PATH"
if [ $? -ne 0 ]; then
  echo "📁 Creating HDFS directory $HDFS_PATH"
  hdfs dfs -mkdir -p "$HDFS_PATH"
fi

echo "⚙️ Generating $GRAPH_TYPE graph with $NUM_NODES nodes..."

EDGELIST=$(python3 graph_generator.py "$NUM_NODES" "$GRAPH_TYPE")
NUM_EDGES=$(echo "$EDGELIST" | grep -c -P '^[^\t]*\t[^\t]*$')

# Check if graph created with edges
if [ "$NUM_EDGES" -lt 1 ]; then
  echo "❌ Ο γράφος έχει $NUM_EDGES ακμές – ακύρωση αποστολής."
  exit 1
fi

echo "$EDGELIST" | hdfs dfs -put - "$HDFS_FULL_PATH"

if [ $? -eq 0 ]; then
  echo "✅ Graph with $NUM_EDGES edges uploaded to $HDFS_FULL_PATH"
else
  echo "❌ Upload failed."
  exit 1
fi
