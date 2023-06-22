#!/bin/bash
#SBATCH --output=sbatch_log/%j.out 
#SBATCH --gres=gpu:4 
#SBATCH --mem=30G
source /usr/itetnas04/data-scratch-01/ddordevic/data/conda/etc/profile.d/conda.sh
conda activate mscenv3
python -u extract_features.py --model swin_s --weights ./model_save/swin_s_ft_contr.pth "$@"