#!/bin/bash
#SBATCH --output=sbatch_log/train_metric_%j.out 
#SBATCH --gres=gpu:4 
#SBATCH --mem=30G
source /usr/itetnas04/data-scratch-01/ddordevic/data/conda/etc/profile.d/conda.sh
conda activate mscenv3
python -u train_metric.py "$@"