import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class EurDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        with open(data_dir + '{}_data.pkl'.format(split), 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, index):
        sents = self.data[index]
        return  sents

    def __len__(self):
        return len(self.data)

def collate_data(batch):

    batch_size = len(batch)
    max_len = max(map(lambda x: len(x), batch))   # get the max length of sentence in current batch
    sents = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x), reverse=True)

    for i, sent in enumerate(sort_by_len):
        length = len(sent)
        sents[i, :length] = sent  # padding the questions

    return  torch.from_numpy(sents)

