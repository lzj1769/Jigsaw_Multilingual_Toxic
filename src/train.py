import config
import dataset
import pandas as pd
import argparse
from torch.utils.tensorboard import SummaryWriter

from model import BERTBaseUncased
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from config import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Training model for Jigsaw Multilingual Toxic Comment Classification')
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers for training. "
                             "Default: 4")
    parser.add_argument("--max-length", type=int, default=512,
                        help="The maximum length of a sequence after tokenizing. "
                             "Default: 512")
    parser.add_argument("--epochs", type=int, default=4,
                        help="Number of epochs used for training. "
                             "Default: 512")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size. Default: 512")

    return parser.parse_args()


def run(num_workers, max_length, epochs, batch_size):
    df1 = pd.read_csv(TRAINING_FILE1, usecols=["comment_text", "toxic"])
    df2 = pd.read_csv(TRAINING_FILE2, usecols=["comment_text", "toxic"])

    df_train = pd.concat([df1, df2], axis=0).reset_index(drop=True)

    df_valid = pd.read_csv(VALIDATION_FILE)

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = dataset.BERTDataset(
        comment_text=df_train.comment_text.values,
        target=df_train.toxic.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=num_workers
    )

    valid_dataset = dataset.BERTDataset(
        comment_text=df_valid.comment_text.values,
        target=df_valid.toxic.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1
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

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    # model = nn.DataParallel(model)
    print(f'num_train_steps = {num_train_steps}')

    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        train_fn(train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = eval_fn(valid_data_loader, model, device)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy Score = {accuracy}")
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy


if __name__ == "__main__":
    seed_everything(seed=42)

    args = parse_args()
    run(num_workers=args.num_workers,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size)
