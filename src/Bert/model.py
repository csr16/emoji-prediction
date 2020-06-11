from transformers import BertModel
import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from data import EmojiDataset, DataLoader, collate_fn

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

class BertClassifier(nn.Module):
    def __init__(self, hidden_size=768, output_size=49, freeze_bert=True):
        super(BertClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # try to freeze bert
        if freeze_bert:
            for p in self.parameters():
                p.requires_grad = False

        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input):
        bert_output = self.bert(input)
        # bert_hidden = bert_output[0]
        # should be in size (batch_size, sent_len, hidden_size)
        final_hidden = bert_output[1]
        # final_hidden = bert_hidden[:, -1, :]
        output = self.linear(final_hidden)
        return output


# model = BertClassifier()
# model = model.to('cuda')
# print(model)
#
# dataset = EmojiDataset()
# dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
# dataiter = iter(dataloader)
# test_sents, test_labels = next(dataiter)
# print(test_sents.shape, test_labels.shape)
#
# test_samples = test_sents.to('cuda')
# pred = model(test_samples)
# print(pred.shape)

