from random import randint
from typing import List

import torch
from torch.utils.data import Dataset

from cfg import alphabet_map


class CavemanGPTDataset(Dataset):
    def __init__(self, texts: List[str], block_size):
        self.block_size = block_size
        tokenized_texts = [[alphabet_map[x] for x in input_text] for input_text in texts]

        self.text_lens = [len(x) for x in tokenized_texts]

        input_ids, labels = zip(*[(x[:-1], x[1:]) for x in tokenized_texts])
        input_ids = list(input_ids)
        labels = list(labels)

        self.pad_ids(block_size, input_ids)
        self.pad_ids(block_size, labels, -100)

        self.input_ids = input_ids
        self.labels = labels # TODO: not ideal that label is always last token

    def pad_ids(self, block_size, input_ids, pad_value=0):
        for idx in range(len(input_ids)):
            if len(input_ids[idx]) < self.block_size:
                padding = [pad_value] * (block_size - len(input_ids[idx]))
                input_ids[idx] += padding
            else:
                input_ids[idx] = input_ids[idx][:self.block_size]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_idx = self.input_ids[idx]
        mask = [0 if x == 0 else 1 for x in input_idx]
        labels = self.labels[idx]

        return {'input_ids': torch.tensor(input_idx),
                'mask': torch.tensor(mask),
                'labels': torch.tensor(labels)}




input_texts = ["abc" * (i+1) for i in range(5)]
ds = CavemanGPTDataset(input_texts, block_size=8)
print(ds[4])