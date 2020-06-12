import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
from transformer.models import Transformer
import emoji
import random
import argparse

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_cuda = torch.cuda.is_available()
print(use_cuda)

if use_cuda:
    torch.cuda.set_device(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

vocab = './data/emoji.dict'
_, src_word2idx, _ = torch.load(vocab)['src_dict']
_, _, tgt_idx2word = torch.load(vocab)['tgt_dict']

def emojilize(opt, model):
    sent_ls = str.split(opt.src)
    sent_len = len(sent_ls)

    tokenlized_sent = []
    for word in sent_ls:
        tokenlized_sent.append(src_word2idx[word])

    sent = torch.LongTensor(tokenlized_sent).view(1,-1)
    sent_len = torch.LongTensor([sent_len]).view(-1)

    with torch.no_grad():
        dec_logits, *_ = model(sent, sent_len)

    # calculate top 5
    maxk = max((1, 5))
    _, predictions_top5 = dec_logits.topk(maxk, 1, True, True)

    emojis_preds = predictions_top5.numpy().tolist()[0]
    print('#############################')
    print('Source Text: {}'.format(opt.src))
    for emoji_id in emojis_preds:
        emoji_pic = emoji.emojize(':'+tgt_idx2word[emoji_id]+':')
        print(emoji_pic)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate emojis from sentences')
    parser.add_argument('-model_path', default='train_log/emoji_model.pt',  type=str, help='Path to the model')
    parser.add_argument('-src', default='conversations that challenge you are the best The ones that make you think and you leave the convo with wisdom', type=str, help='sentence for translating into emojis')
    parser.add_argument('-data_path', default='data/emoji-train.t7', help='Path to the preprocessed data')

    # network params
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-d_ff', type=int, default=2048)
    parser.add_argument('-n_heads', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-share_proj_weight', action='store_true')
    parser.add_argument('-share_embs_weight', action='store_true')
    parser.add_argument('-weighted_model', action='store_true')

    # training params
    parser.add_argument('-lr', type=float, default=0.002)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-max_src_seq_len', type=int, default=50)
    parser.add_argument('-max_tgt_seq_len', type=int, default=10)
    parser.add_argument('-max_grad_norm', type=float, default=None)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)
    parser.add_argument('-display_freq', type=int, default=100)
    parser.add_argument('-log', default=None)

    opt = parser.parse_args()

    data = torch.load(opt.data_path)
    opt.src_vocab_size = len(data['src_dict'])
    opt.tgt_vocab_size = len(data['tgt_dict'])

    print('Creating new model parameters..')
    model = Transformer(opt)  # Initialize a model state.
    model_state = {'opt': opt, 'curr_epochs': 0, 'train_steps': 0}

    print('Reloading model parameters..')
    model_state = torch.load('./train_log/emoji_model.pt', map_location=device)
    model.load_state_dict(model_state['model_params'])

    emojilize(opt, model)