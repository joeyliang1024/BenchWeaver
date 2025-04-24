#!/bin/bash
export JAVA_HOME="/usr/lib/java"

# Set the directory (can be modified or passed as an argument)
TASK_NAME="taide-bench"
DIRECTORY="/work/u5110390/BenchWeaver/config/main_pipeline/$TASK_NAME"
LOG_DIR="/work/u5110390/BenchWeaver/logs/main_pipeline/$TASK_NAME"

# Check if input directory exists
if [ ! -d "$DIRECTORY" ]; then
  echo "Directory does not exist."
  exit 1
fi

# Check if log directory exists, if not create it
if [ ! -d "$LOG_DIR" ]; then
  echo "Log directory does not exist. Creating it."
  mkdir -p "$LOG_DIR"
fi

# Loop through all files in the directory
for FILE in "$DIRECTORY"/*; do
  if [ -f "$FILE" ]; then
    echo "Loading file: $FILE"
    FILENAME=$(basename "$FILE" .yaml)
    # Execute the command with the file and log output to a .log file
    bench-weaver-cli eval --config "$FILE" 2>&1 | tee "$LOG_DIR/$FILENAME.log"
  fi
done
