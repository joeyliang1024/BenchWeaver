#!/bin/bash

TASK="gpqa"
MODE="opqa"
PIPELINE="same"
CONFIG_PATH="/home/joeyliang/BenchWeaver/config/gpqa/opqa.yaml"

CUDA_VISIBLE_DEVICES=1,2 bench-weaver-cli eval \
    --task $TASK \
    --mode $MODE \
    --pipeline $PIPELINE \
    --config $CONFIG_PATH