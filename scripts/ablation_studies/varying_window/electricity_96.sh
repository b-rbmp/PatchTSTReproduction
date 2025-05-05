#!/bin/bash
#SBATCH --job-name=patchtst             # Job name
#SBATCH --partition=gpu                   # Partition (queue) to submit to
#SBATCH --gres=gpu:1                      # Request 1 GPU (full GPU)
#SBATCH --ntasks=1                        # Run a single task (1 instance of your program)
#SBATCH --cpus-per-task=16                 # Number of CPU cores per task (adjust based on your needs)
#SBATCH --mem=64G                         # Total memory (RAM) for the job (adjust based on your dataset)
#SBATCH --time=72:00:00                    # Time limit (24 hours)
#SBATCH --output=patchtst_%j.log               # Standard output and error log (%j is replaced by job ID)
#SBATCH --constraint=h100

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/ablation_window" ]; then
    mkdir ./logs/ablation_window
fi

model_name=PatchTST
model_identifier=patchtst_electricity
dataset=electricity
prediction_length=96
for input_length in 24 48 96 192 336 720
do
    python -u src/training/supervised/train.py \
      --train_mode \
      --model_identifier $model_identifier'_'$input_length'_'$prediction_length'_'$dataset \
      --model $model_name \
      --dataset $dataset \
      --features M \
      --input_length $input_length \
      --prediction_length $prediction_length \
      --encoder_input_size 321 \
      --num_encoder_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_fcn 256 \
      --dropout 0.2 \
      --fc_dropout 0.2 \
      --head_dropout 0 \
      --patch_length 16 \
      --stride 8 \
      --epochs 100 \
      --patience 10 \
      --learning_rate_adjustment TST \
      --lr_pct_start 0.2 \
      --fp16 \
      --bootstrap_iterations 1 --batch_size 1 --learning_rate 0.0001 >logs/ablation_window/$model_identifier'_'$input_length'_'$prediction_length.log 
done