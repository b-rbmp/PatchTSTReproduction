# PatchTST Reproduction (PyTorch)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PyTorch reproduction of **PatchTST: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers** (ICLR 2023).

- **Original Paper:** [https://openreview.net/forum?id=Jbdc0vTOcol](https://openreview.net/forum?id=Jbdc0vTOcol)
- **Reproduction Report:** See `report.pdf` in this repository for detailed methods, results, and discussion.
- **Official Code:** [https://github.com/yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST)

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)

## Prerequisites

- Python 3.8+
- PyTorch
- NVIDIA GPU + CUDA (Recommended)
- Libraries listed in `requirements.txt` (run `pip install -r requirements.txt`)

## Installation

1.  **Clone:**
    ```bash
    git clone https://github.com/b-rbmp/PatchTST-Reproduction.git
    cd PatchTST-Reproduction
    ```
2.  **Setup Environment:**
    ```bash
    conda create -n patchtst_rep python=3.9
    conda activate patchtst_rep
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Preparation

1.  Create a `./datasets/` directory in the repository root.
2.  Download benchmark datasets (ETT, Electricity, Traffic, Weather, ILI) and place them in the appropriate subdirectories within `./datasets/` (e.g., `./datasets/ETT-small/`, `./datasets/weather/`).
    * Download here: [Benchmark Datasets Link](https://drive.google.com/file/d/1phIAd-QenHxTPD2wC3TVVXrjdj8mQT3C/view?usp=sharing) 

## Usage

Training and evaluation are performed using scripts under `src/training/`. Configuration is controlled via command-line arguments (see `src/training/*/config.py`).

**Example Scripts:** The `scripts/` directory contains detailed examples for various modes:

-   `scripts/supervised/`: Standard supervised forecasting.
-   `scripts/self_supervised/`: Masked pre-training followed by fine-tuning or linear probing.
-   `scripts/transfer_learning/`: Pre-training on one dataset, applying to another.
-   `scripts/ablation_studies/`: Experiments testing model components (patching, channel-independence, look-back window).

**Basic Supervised Training Example (ETTh1):**
```bash
python -u src/training/supervised/train.py \
  --train_mode \
  --model_identifier patchtst_etth1_512_96 \
  --model PatchTST \
  --dataset etth1 \
  --features M \
  --input_length 512 \
  --prediction_length 96 \
  --encoder_input_size 7 \
  --num_encoder_layers 3 \
  --n_heads 4 \
  --d_model 16 \
  --d_fcn 128 \
  --dropout 0.3 \
  --fc_dropout 0.3 \
  --head_dropout 0 \
  --patch_length 16 \
  --stride 8 \
  --epochs 100 \
  --patience 20 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --bootstrap_iterations 5 \
  --use_cuda True \
  --checkpoint_dir ./checkpoints/