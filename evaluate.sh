#!/bin/bash
#SBATCH --output=sbatch_log/%j.out 
#SBATCH --gres=gpu:4 
#SBATCH --mem=30G
source /usr/itetnas04/data-scratch-01/ddordevic/data/conda/etc/profile.d/conda.sh
conda activate mscenv3
python -u evaluate.py --model vit_s16 --train ./CIFAR-10/vit_s16_cifar10_train_contr --test ./CIFAR-10/vit_s16_cifar10_test_contr --K 1 10 100 1000 --savedir ./CIFAR-10/ "$@"