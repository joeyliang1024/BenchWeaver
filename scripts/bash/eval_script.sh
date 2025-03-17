#!/bin/bash

CONFIG_PATH="/home/joeyliang/BenchWeaver/config/evaluation.yaml"

CUDA_VISIBLE_DEVICES=1,2 bench-weaver-cli eval --config $CONFIG_PATH