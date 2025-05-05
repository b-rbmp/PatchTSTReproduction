#!/bin/bash
#SBATCH --job-name=patchtst             # Job name
#SBATCH --partition=gpu                   # Partition (queue) to submit to
#SBATCH --gres=gpu:1                      # Request 1 GPU (full GPU)
#SBATCH --ntasks=1                        # Run a single task (1 instance of your program)
#SBATCH --cpus-per-task=16                 # Number of CPU cores per task (adjust based on your needs)
#SBATCH --mem=64G                         # Total memory (RAM) for the job (adjust based on your dataset)
#SBATCH --time=24:00:00                    # Time limit (24 hours)
#SBATCH --output=patchtst_%j.log               # Standard output and error log (%j is replaced by job ID)

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/self_supervised" ]; then
    mkdir ./logs/self_supervised
fi

if [ ! -d "./logs/self_supervised/supervised" ]; then
    mkdir ./logs/self_supervised/supervised
fi

model_name=PatchTST
model_identifier=weather_supervised
dataset=weather
input_length=512


# Define prediction length:
# 96, 192, 336, 720
for prediction_length in 96 192 336 720
do
    python -u src/training/self_supervised/supervised_bootstrap.py \
        --model_identifier $model_identifier'_'$input_length'_'$prediction_length \
        --model $model_name \
        --dataset $dataset \
        --features M \
        --input_length $input_length \
        --prediction_length $prediction_length \
        --patch_length 12 \
        --stride 12 \
        --epochs 100 \
        --bootstrap_iterations 3 --batch_size 64 >logs/self_supervised/supervised/$model_identifier'_'$input_length'_'$prediction_length.log 
done