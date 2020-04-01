""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

import dataset
import pandas as pd
import argparse
import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import logging

from model import BERTBaseUncased
from config import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Training model for Jigsaw Multilingual Toxic Comment Classification')
    parser.add_argument("--num-workers", type=int, default=12,
                        help="Number of workers for training. "
                             "Default: 24")
    parser.add_argument("--max-length", type=int, default=512,
                        help="The maximum length of a sequence after tokenizing. "
                             "Default: 512")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of epochs used for training. "
                             "Default: 2")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size. Default: 512")

    return parser.parse_args()


def run(num_workers, max_length, epochs, batch_size):
    df1 = pd.read_csv(TRAINING_FILE1, usecols=["comment_text", "toxic"])
    df2 = pd.read_csv(TRAINING_FILE2, usecols=["comment_text", "toxic"])

    df_train = pd.concat([df1, df2], axis=0).reset_index(drop=True)

    df_valid = pd.read_csv(VALIDATION_FILE)

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    tokenizer = transformers.BertTokenizer.from_pretrained(
        PRETRAIN_BERT_PATH,
        do_lower_case=True
    )

    train_dataset = dataset.BERTDataset(
        comment_text=df_train.comment_text.values,
        target=df_train.toxic.values,
        tokenizer=tokenizer,
        max_len=max_length
    )

    valid_dataset = dataset.BERTDataset(
        comment_text=df_valid.comment_text.values,
        target=df_valid.toxic.values,
        tokenizer=tokenizer,
        max_len=max_length
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    device = torch.device("cuda")
    model = BERTBaseUncased()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    num_train_steps = int(len(df_train) / batch_size * epochs)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    # model = nn.DataParallel(model)
    print(f'num_train_steps = {num_train_steps}')

    best_accuracy = 0
    for epoch in range(epochs):
        train_fn(train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = eval_fn(valid_data_loader, model, device)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy Score = {accuracy}")
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_file_1",
        default=None,
        type=str,
        required=True,
        help="The input training data file from Toxic Comment Classification Challenge 2018."
    )

    parser.add_argument(
        "--train_data_file_2",
        default=None,
        type=str,
        required=True,
        help="The input training data file from Jigsaw Unintended Bias in Toxicity Classification 2019."
    )

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model",
    )

    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")

    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=16, type=int, help="Batch size per GPU/CPU for evaluation.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )

    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )

    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )


if __name__ == "__main__":
    seed_everything(seed=42)

    args = parse_args()
    run(num_workers=args.num_workers,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size)
