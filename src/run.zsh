#!/usr/local_rwth/bin/zsh

#SBATCH -J run
#SBATCH -o run.txt

#SBATCH -t 60:00:00 --mem=60G
#SBATCH --gres=gpu:2 -A rwth0455 -c 16

module load cuda
source ~/.zshrc
source ~/miniconda3/bin/activate kaggle

data_dir=/home/rwth0455/kaggle/Jigsaw_Multilingual_Toxic/input
output_dir=/work/rwth0455/kaggle/Jigsaw_Multilingual_Toxic/output
log_dir=/home/rs619065/kaggle/Jigsaw_Multilingual_Toxic/log

python run.py \
--data_dir $data_dir \
--model_type bert \
--model_name_or_path bert-base-multilingual-uncased \
--output_dir $output_dir/bert \
--log_dir $log_dir \
--do_train \
--max_seq_length 256 \
--num_train_epochs 3 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 3e-05 \
--weight_decay 0.0001 \
--num_workers 12 \
--do_lower_case \
--save_steps 10000 \
--logging_steps 100 \
--overwrite_output_dir \
--evaluate_during_training
