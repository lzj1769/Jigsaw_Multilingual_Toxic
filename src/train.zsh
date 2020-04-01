#!/usr/local_rwth/bin/zsh

#SBATCH -J train
#SBATCH -o train.txt

#SBATCH -t 60:00:00 --mem=60G
#SBATCH --gres=gpu:1 -A rwth0455 -c 12

module load cuda

source ~/.zshrc
source ~/miniconda3/bin/activate kaggle

input_dir=/home/rwth0455/kaggle/Jigsaw_Multilingual_Toxic/input
output_dir=/work/rwth0455/kaggle/Jigsaw_Multilingual_Toxic/output

train_data_file_1=$input_dir/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv
train_data_file_2=$input_dir/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv
test_file=$input/jigsaw-multilingual-toxic-comment-classification/test.csv
validation_file=$input/jigsaw-multilingual-toxic-comment-classification/validation.csv
sample_file=$input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv

python train.py \
--train_data_file_1 $train_data_file_1 \
--train_data_file_2 $train_data_file_2
