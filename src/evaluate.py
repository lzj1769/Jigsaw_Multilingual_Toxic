import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from sklearn import metrics
from utils import load_data


def evaluate(args, eval_dataset, model):
    results = {}

    args.eval_batch_size = args.per_gpu_eval_batch_size * args.n_gpu
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 num_workers=args.num_workers)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    eval_loss = 0.0
    nb_eval_steps = 0
    y_score = None
    y_true = None

    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[3]}

            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1
        if y_score is None:
            y_score = logits.detach().cpu().numpy()
            y_true = inputs["labels"].detach().cpu().numpy()
        else:
            y_score = np.append(y_score, logits.detach().cpu().numpy(), axis=0)
            y_true = np.append(y_true, inputs["labels"].detach().cpu().numpy(), axis=0)

    results['loss'] = eval_loss / nb_eval_steps
    results['roc'] = metrics.roc_auc_score(y_true=y_true, y_score=y_score[:, 1])

    return results
