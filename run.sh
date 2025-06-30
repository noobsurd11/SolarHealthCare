#!/bin/bash

# Activate environment if needed
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env_name

# Set timestamp
now=$(date +"%Y%m%d_%H%M")

# Define experiment name and config
EXP_NAME="model_run_${now}"
CONFIG="configs/defaults.yaml"
DEVICE="cuda"
MODE='train'

# Create logs directory if needed
mkdir -p logs

# Run training
echo "[🚀] Starting experiment: $EXP_NAME"
python main.py \
  --mode $MODE \
  --config $CONFIG \
  --device $DEVICE \
  --experiment $EXP_NAME \
  > logs/train_$EXP_NAME.log 2>&1

echo "Log saved to logs/train_$EXP_NAME.log"
