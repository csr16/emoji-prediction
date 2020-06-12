from __future__ import print_function
import os
import sys
import time
import math
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm
import torch.optim as optim

from data import data_utils
from data.data_utils import load_train_data
from transformer.models import Transformer
from transformer.optimizer import ScheduledOptimizer

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
    print('Loading training and development data..')
    _, _, train_iter, dev_iter = load_train_data(opt.data_path, opt.batch_size,
                                                 opt.max_src_seq_len, opt.max_tgt_seq_len, use_cuda)
    # Create a new model or load an existing one.
    model, model_state = create_model(opt)
    init_epoch = model_state['curr_epochs']
    if init_epoch >= opt.max_epochs:
        print('Training is already complete.',
              'current_epoch:{}, max_epoch:{}'.format(init_epoch, opt.max_epochs))
        sys.exit(0)

    # Loss and Optimizer
    # If size_average=True (default): Loss for a mini-batch is averaged over non-ignore index targets.
    criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=data_utils.PAD)
    optimizer = ScheduledOptimizer(optim.Adam(model.trainable_params(), betas=(0.9, 0.98), eps=1e-9),
                                   opt.d_model, opt.n_layers, opt.n_warmup_steps)
    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_dev_file = opt.log + '.valid.log'
        if not os.path.exists(log_train_file) and os.path.exists(log_dev_file):
            with open(log_train_file, 'w') as log_tf, open(log_dev_file, 'w') as log_df:
                log_tf.write('epoch,ppl,sents_seen\n')
                log_df.write('epoch,ppl,sents_seen\n')
        print('Training and validation log will be written in {} and {}'
              .format(log_train_file, log_dev_file))

    train_ls, train_acc, eval_ls, eval_acc = [], [], [], []
    best_acc = 0.0
    for epoch in range(init_epoch + 1, opt.max_epochs + 1):
        # Execute training steps for 1 epoch.
        epoch_train_ls, epoch_train_acc = train(model, criterion, optimizer, train_iter, model_state)
        print('Epoch {}'.format(epoch), 'Train_loss: {0:.4f}'.format(epoch_train_ls),
              'Train_acc: {0:.3f}'.format(epoch_train_acc))
        train_ls.append(epoch_train_ls)
        train_acc.append(epoch_train_acc)
        # Execute a validation step.
        epoch_eval_ls, epoch_eval_acc = eval(model, criterion, dev_iter)
        print('Epoch {}'.format(epoch), 'Eval_loss: {0:.4f}'.format(epoch_eval_ls),
              'Eval_acc: {0:.3f}'.format(epoch_eval_acc))
        eval_ls.append(epoch_eval_ls)
        eval_acc.append(epoch_eval_acc)

        # Save the model checkpoint in every 1 epoch.
        if epoch_eval_acc > best_acc or best_acc == 0.0:
            model_state['curr_epochs'] += 1
            model_state['model_params'] = model.state_dict()
            torch.save(model_state, opt.model_path)
            best_acc = epoch_eval_acc
            print('The model checkpoint file has been saved')

    # plot curve of ls and acc
    x = range(0, len(train_ls))
    plt.figure()
    plt.title('train and valid loss of Transformer')
    plt.xlabel(u'epochs')
    plt.ylabel(u'train and valid loss')
    plt.plot(x, train_ls, label='training loss')
    plt.scatter(train_ls.index(min(train_ls)), min(train_ls), label='best training loss = {:.4f}'.format(min(train_ls)))
    plt.plot(x, eval_ls, label='validation loss')
    plt.scatter(eval_ls.index(min(eval_ls)), min(eval_ls), label='best validation loss = {:.4f}'.format(min(eval_ls)))
    plt.legend()
    plt.savefig('./train and valid loss curves of Transformer')

    print('best loss on train set: ', min(train_ls))
    print('best loss on val set: ', min(eval_ls))

    plt.figure()
    plt.title('train and valid acc of Transformer')
    plt.xlabel(u'epochs')
    plt.ylabel(u'train and valid acc')
    plt.plot(x, train_acc, label='training acc')
    plt.scatter(train_acc.index(max(train_acc)), max(train_acc), label='best training acc = {:.4f}'.format(max(train_acc)))
    plt.plot(x, eval_acc, label='validation acc')
    plt.scatter(eval_acc.index(max(eval_acc)), max(eval_acc), label='best validation acc = {:.4f}'.format(max(eval_acc)))
    plt.legend()
    plt.savefig('./train and valid acc curves of Transformer')

    print('best acc on train set: ', max(eval_acc))
    print('best acc on val set: ', max(eval_acc))




def train(model, criterion, optimizer, train_iter, model_state):  # TODO: fix opt
    model.train()
    opt = model_state['opt']
    train_loss, train_loss_total = 0.0, 0.0
    total_correct = 0
    n_words, n_words_total = 0, 0
    n_sents, n_sents_total = 0, 0
    start_time = time.time()

    for batch_idx, batch in enumerate(train_iter):
        enc_inputs, enc_inputs_len = batch.src
        dec_, dec_inputs_len = batch.trg
        dec_targets = dec_[:, 1]
        dec_inputs_len = dec_inputs_len - 2

        # Execute a single training step: forward
        optimizer.zero_grad()
        dec_logits, _ = model(enc_inputs, enc_inputs_len)
        _, predictions = torch.max(dec_logits, 1)
        step_loss = criterion(dec_logits, dec_targets.contiguous().view(-1))

        # Execute a single training step: backward
        step_loss.backward()
        if opt.max_grad_norm:
            clip_grad_norm(model.trainable_params(), float(opt.max_grad_norm))
        optimizer.step()
        optimizer.update_lr()
        model.proj_grad()  # works only for weighted transformer
        # print(step_loss)
        train_loss_total += step_loss.item()
        total_correct += torch.sum(predictions == dec_targets.data)
        n_words_total += torch.sum(dec_inputs_len)
        n_sents_total += dec_inputs_len.size(0)  # batch_size
        model_state['train_steps'] += 1

    # return per_word_loss over 1 epoch
    return train_loss_total / n_words_total.item(), total_correct.double().item() / n_words_total.item()


def eval(model, criterion, dev_iter):
    model.eval()
    eval_loss_total = 0.0
    total_correct = 0
    n_words_total, n_sents_total = 0, 0

    print('Evaluation')
    with torch.no_grad():
        for batch_idx, batch in enumerate(dev_iter):
            enc_inputs, enc_inputs_len = batch.src
            dec_, dec_inputs_len = batch.trg
            dec_targets = dec_[:, 1]
            dec_inputs_len = dec_inputs_len - 2

            dec_logits, *_ = model(enc_inputs, enc_inputs_len)
            _, predictions = torch.max(dec_logits, 1)
            step_loss = criterion(dec_logits, dec_targets.contiguous().view(-1))
            eval_loss_total += step_loss.item()
            total_correct += torch.sum(predictions == dec_targets.data)
            n_words_total += torch.sum(dec_inputs_len)
            n_sents_total += dec_inputs_len.size(0)

    # return per_word_loss
    return eval_loss_total / n_words_total.item(), total_correct.double().item() / n_words_total.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Hyperparams')
    # data loading params
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
    parser.add_argument('-max_epochs', type=int, default=100)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-max_src_seq_len', type=int, default=50)
    parser.add_argument('-max_tgt_seq_len', type=int, default=10)
    parser.add_argument('-max_grad_norm', type=float, default=None)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)
    parser.add_argument('-display_freq', type=int, default=100)
    parser.add_argument('-log', default=None)
    parser.add_argument('-model_path', type=str, default='train_log/emoji_model.pt')

    opt = parser.parse_args()
    print(opt)
    main(opt)
    print('Terminated')
