#!/bin/bash
#SBATCH --output=sbatch_log/evaluate_cifar10_%j.out 
#SBATCH --gres=gpu:4 
#SBATCH --mem=30G
source /usr/itetnas04/data-scratch-01/ddordevic/data/conda/etc/profile.d/conda.sh
conda activate mscenv3
python -u evaluate_cifar10.py "$@"