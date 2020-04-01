#!/usr/local_rwth/bin/zsh

#SBATCH -J train
#SBATCH -o train.txt

#SBATCH -t 60:00:00 --mem=50G
#SBATCH --gres=gpu:1 -A rwth0455

module load cuda

source ~/.zshrc
source ~/miniconda3/bin/activate kaggle

python train.py
