# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import argparse
import os

from transformer.models import Transformer
from data.data_utils import load_train_data
from data.data_utils import convert_idx2text
from transformer.translator import Translator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
print(use_cuda)

if use_cuda:
    torch.cuda.set_device(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

def create_model(opt):
    data = torch.load(opt.data_path)
    opt.src_vocab_size = len(data['src_dict'])
    opt.tgt_vocab_size = len(data['tgt_dict'])

    print('Creating new model parameters..')
    model = Transformer(opt)  # Initialize a model state.
    model_state = {'opt': opt, 'curr_epochs': 0, 'train_steps': 0}

    # If opt.model_path exists, load model parameters.
    if os.path.exists(opt.model_path):
        print('Reloading model parameters..')
        model_state = torch.load(opt.model_path)
        model.load_state_dict(model_state['model_params'])

    if use_cuda:
        print('Using GPU..')
        model = model.to(device)

    return model, model_state

def main(opt):
    _, _, train_iter, test_iter = load_train_data(opt.data_path, opt.batch_size,
                                                 opt.max_src_seq_len, opt.max_tgt_seq_len, use_cuda)
    model, model_state = create_model(opt)
    _, _, tgt_idx2word = torch.load(opt.vocab)['tgt_dict']

    lines = 0
    print('Translated output will be written in {}'.format(opt.decode_output))
    total_correct = 0
    top5_total_correct = 0
    n_words_total = 0
    with open(opt.decode_output, 'w') as output:
        with torch.no_grad():
            for batch in test_iter:
                enc_inputs, enc_inputs_len = batch.src
                dec_, dec_inputs_len = batch.trg
                dec_targets = dec_[:, 1]
                dec_inputs_len = dec_inputs_len - 2
                dec_logits, *_ = model(enc_inputs, enc_inputs_len)
                _, predictions = torch.max(dec_logits, 1)
                total_correct += torch.sum(predictions == dec_targets.data)

                n_words_total += torch.sum(dec_inputs_len)

                # top 5
                maxk = max((1, 5))
                dec_targets = dec_targets.view(-1, 1)
                _, predictions_top5 = dec_logits.topk(maxk, 1, True, True)
                # print(dec_targets.shape, predictions_top5.shape)
                top5_total_correct += torch.eq(predictions_top5, dec_targets).sum().float().item()

    return total_correct.double().item() / n_words_total.item(), top5_total_correct / n_words_total.item()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translation hyperparams')
    parser.add_argument('-model_path', default='train_log/emoji_model.pt',  type=str, help='Path to the model')
    parser.add_argument('-data_path', default='data/emoji-train.t7', help='Path to the preprocessed data')
    parser.add_argument('-vocab', required=True, type=str, help='Path to an existing vocabulary file')
    parser.add_argument('-input', required=True, type=str, help='Path to the source file to translate')
    parser.add_argument('-decode_output', required=True, type=str, help='Path to write translated sequences' )

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
    parser.add_argument('-max_epochs', type=int, default=100)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-max_src_seq_len', type=int, default=50)
    parser.add_argument('-max_tgt_seq_len', type=int, default=10)
    parser.add_argument('-max_grad_norm', type=float, default=None)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)
    parser.add_argument('-display_freq', type=int, default=100)
    parser.add_argument('-log', default=None)

    opt = parser.parse_args()
    print(opt)
    top1_acc, top5_acc = main(opt)
    print('Terminated')

    print('top1 acc: ', top1_acc)
    print('top5 acc: ', top5_acc)