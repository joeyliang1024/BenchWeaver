#!/bin/bash
# export JAVA_HOME="/usr/lib/java"
export CUDA_VISIBLE_DEVICES="0,1"

CONFIG_PATH="../../config/evaluation.yaml"

bench-weaver-cli eval --config "$CONFIG_PATH"