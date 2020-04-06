import torch
from torch.utils.data import Dataset


class BERTDataset(Dataset):
    def __init__(self, comment_text, labels, tokenizer, max_length):
        self.comment_text = comment_text
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, item):
        comment_text = str(self.comment_text[item])

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True,
            return_attention_mask=True
        )

        input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)
        token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long)
        labels = torch.tensor(self.labels[item], dtype=torch.long)

        assert len(input_ids) == self.max_length, "Error with input length {} vs {}".format(
            len(input_ids), self.max_length
        )
        assert len(attention_mask) == self.max_length, "Error with input length {} vs {}".format(
            len(attention_mask), self.max_length
        )
        assert len(token_type_ids) == self.max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), self.max_length
        )

        return [input_ids, attention_mask, token_type_ids, labels]
