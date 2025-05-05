#!/bin/bash
#SBATCH --job-name=patch_tst_             # Job name
#SBATCH --partition=gpu                   # Partition (queue) to submit to
#SBATCH --gres=gpu:1                      # Request 1 GPU (full GPU)
#SBATCH --ntasks=1                        # Run a single task (1 instance of your program)
#SBATCH --cpus-per-task=16                 # Number of CPU cores per task (adjust based on your needs)
#SBATCH --mem=64G                         # Total memory (RAM) for the job (adjust based on your dataset)
#SBATCH --time=40:00:00                    # Time limit (24 hours)
#SBATCH --output=patchtst_%j.log               # Standard output and error log (%j is replaced by job ID)

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/supervised" ]; then
    mkdir ./logs/supervised
fi

model_name=PatchTST
model_identifier=patchtst_weather
dataset=weather
input_length=512

for prediction_length in 96 192 336 720
do
    python -u src/training/supervised/train.py \
      --train_mode \
      --model_identifier $model_identifier'_'$input_length'_'$prediction_length \
      --model $model_name \
      --dataset $dataset \
      --features M \
      --input_length $input_length \
      --prediction_length $prediction_length \
      --encoder_input_size 21 \
      --num_encoder_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_fcn 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_length 16\
      --stride 8 \
      --epochs 100 \
      --patience 20 \
      --bootstrap_iterations 5 --batch_size 128 --learning_rate 0.0001 >logs/supervised/$model_identifier'_'$input_length'_'$prediction_length.log 
done