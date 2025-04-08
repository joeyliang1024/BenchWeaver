#!/bin/bash


# Set variables (you can edit these as needed)
RESULT_DIR="score/trans_template_exp/cmmlu/few_shot"
EXP_NAME=$(echo "$RESULT_DIR" | awk -F'/' '{print $(NF-1) "/" $NF}')
EXPORT_DIR="score/translation_results/$EXP_NAME"
CHECK_MODEL_NAME="gpt-4o"

# Run the Python script
python ../eval_translation.py \
  --result_dir="$RESULT_DIR" \
  --export_dir="$EXPORT_DIR" \
  --check_model_name="$CHECK_MODEL_NAME" \
  # --test_mode="true"
