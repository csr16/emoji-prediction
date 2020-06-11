import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

class EmojiDataset(Dataset):
    def __init__(self,
                 X_data_file = '../data/bert_sentences.npy',
                 y_data_file = '../data/filtered_labels.npy'):
        self.X_data = np.load(X_data_file)
        self.y_data = np.load(y_data_file)

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        sentence = self.X_data[idx]
        label = random.choice(self.y_data[idx])
        return torch.tensor(sentence), label

def collate_fn(batch_data):
    # batch_data 为一个batch的数据组成的列表，data中某一元素的形式如下
    # (tensor([1, 2, 3, 5]), 4, 0)
    # 后续将填充好的序列数据输入到RNN模型时需要使用pack_padded_sequence函数
    # pack_padded_sequence函数要求要按照序列的长度倒序排列
    batch_data.sort(key=lambda xi: len(xi[0]), reverse=True)
    data_length = [len(xi[0]) for xi in batch_data]
    sent_seq = [xi[0] for xi in batch_data]
    label = [xi[1] for xi in batch_data]
    padded_sent_seq = pad_sequence(sent_seq, batch_first=True, padding_value=0)
    # return padded_sent_seq, data_length, torch.tensor(label, dtype=torch.float32)
    return padded_sent_seq, torch.tensor(label, dtype=torch.long)


# dataset = EmojiDataset()
# print(dataset[0])
# print(dataset[1])
# print(dataset[2])
#
# dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
# count = 0
# for sents, labels in dataloader:
#     count += 1
#     if count == 5:
#         break
#     print("###############NEW BATCH##############")
#     print(sents)
#     print(labels)
