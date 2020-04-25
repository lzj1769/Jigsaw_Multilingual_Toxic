import os
import random
import pandas as pd
import numpy as np
import torch
import dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_data(args, tokenizer, data):
    if data == "train":
        training_file_1 = os.path.join(args.data_dir,
                                       "jigsaw-multilingual-toxic-comment-classification",
                                       "jigsaw-toxic-comment-train.csv")
        training_file_2 = os.path.join(args.data_dir,
                                       "jigsaw-multilingual-toxic-comment-classification",
                                       "jigsaw-unintended-bias-train.csv")
        df_train1 = pd.read_csv(training_file_1)
        df_train2 = pd.read_csv(training_file_2)

        df_train2.toxic = (df_train2.toxic > 0).astype(int)

        # Combine df_train1 with a subset of df_train2
        df_train = pd.concat([
            df_train1[['comment_text', 'toxic']],
            df_train2[['comment_text', 'toxic']]
        ])

        train_dataset = dataset.BERTDataset(
            comment_text=df_train.comment_text.values,
            labels=df_train.toxic.values,
            tokenizer=tokenizer,
            max_length=args.max_seq_length
        )

        return train_dataset

    elif data == "validation":
        # validation_file = os.path.join(args.data_dir,
        #                                "jigsaw-multilingual-toxic-comment-classification",
        #                                "validation.csv")
        validation_file = os.path.join(args.data_dir,
                                       "translated",
                                       "jigsaw_miltilingual_valid_translated.csv")

        df_valid = pd.read_csv(validation_file)

        valid_dataset = dataset.BERTDataset(
            comment_text=df_valid.translated.values,
            labels=df_valid.toxic.values,
            tokenizer=tokenizer,
            max_length=args.max_seq_length
        )

        return valid_dataset
