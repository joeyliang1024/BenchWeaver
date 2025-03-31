#!/bin/bash
export JAVA_HOME="/usr/lib/java"

CONFIG_PATH="../../config/trans_template_exp/tmmlu.yaml"

bench-weaver-cli eval --config $CONFIG_PATH 2>&1 | tee output.log