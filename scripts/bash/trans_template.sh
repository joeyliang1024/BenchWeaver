#!/bin/bash
export JAVA_HOME="/usr/lib/java"

CONFIG_PATH="/work/u5110390/BenchWeaver/config/trans_template_exp/cmmlu.yaml"

bench-weaver-cli eval --config $CONFIG_PATH 2>&1 | tee cmmlu.log