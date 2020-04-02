import transformers
import torch.nn as nn
import torch


class BERTBaseUncased(nn.Module):
    def __init__(self, pretrained_model_name_or_path):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(pretrained_model_name_or_path)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768 * 2, 1)

    def forward(self, ids, mask, token_type_ids):
        o1, _ = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        mean_pooling = torch.mean(o1, 1)
        max_pooling, _ = torch.max(o1, 1)
        cat = torch.cat((mean_pooling, max_pooling), 1)
        bo = self.bert_drop(cat)
        output = self.out(bo)

        return output
