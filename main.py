import numpy as np
import pickle
import random
import torch
import emoji
import torch.nn as nn
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence
from data import preprocess, EmojiDataset, collate_fn, plotdata
from model import Torchmoji
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR

parser = argparse.ArgumentParser(description='PyTorch emoji prediction model')
parser.add_argument('--model_type', type=str, default='baseline')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=32, metavar='N',
                    help='eval batch size')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, help='GPU device id used')

args = parser.parse_args()

if args.model_type == 'baseline':
    # data preprocess and prepare
    data_path = './data/dev.txt'
    split_ratio = 0.3
    preprocess(data_path, split_ratio)

    # dataset load and plot
    train_dataset = EmojiDataset('./data/Xtrain.npy', './data/ytrain.npy')
    plotdata(np.load('./data/Xtrain.npy', allow_pickle = True), np.load('./data/ytrain.npy', allow_pickle = True))
    test_dataset = EmojiDataset('./data/Xtest.npy', './data/ytest.npy')
    train_dataloader = DataLoader(train_dataset, batch_size = args.train_batch_size, shuffle = False, collate_fn = collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size = args.eval_batch_size, shuffle = False, collate_fn = collate_fn)

    torch.manual_seed(args.seed)


    # model prepare
    use_gpu = False

    if use_gpu:
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(args.gpu_id)
    else:
        device = torch.device("cpu")

    with open("./data/word2id.pickle", "rb+") as f:
        voc = pickle.load(f)
    with open("./data/label2id.pickle", "rb+") as f:
        emojis = pickle.load(f)
    with open("./data/id2word.pickle", "rb+") as f:
        id2word = pickle.load(f)
    with open("./data/id2label.pickle", "rb+") as f:
        id2label = pickle.load(f)

    nb_tokens = len(voc.keys())
    nb_classes =len(emojis.keys())
    lr = 0.001

    emojimodel = Torchmoji(nb_classes, nb_tokens)
    emojimodel = emojimodel.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(emojimodel.parameters(), lr=lr)

# evaluate on dataset
def evaluate():
    emojimodel.eval()
    losses, accs = [], []
    with torch.no_grad():
        for sents, data_len, labels in test_dataloader:
            sents = sents.to(device)
            data_len = data_len.to(device)
            labels = labels.to(device)

            output = emojimodel(sents, data_len)
            loss = criterion(output, labels)
            res = output.argmax(dim=1).cpu().numpy()
            gt = labels.cpu().numpy()
            correct_count = (res == gt).sum()
            acc = correct_count / gt.shape[0]
            losses.append(loss.item())
            accs.append(acc)
    return losses, accs

# train on dataset
def train():
    emojimodel.train()
    losses, accs = [], []
    for sents, data_len, labels in train_dataloader:
        sents = sents.to(device)
        data_len = data_len.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = emojimodel(sents, data_len)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        res = output.argmax(dim=1).cpu().numpy()
        gt = labels.cpu().numpy()
        correct_count = (res == gt).sum()
        acc = correct_count / gt.shape[0]
        losses.append(loss.item())
        accs.append(acc)
    return losses, accs

# start training and plot the result
def Train():
    total_train_losses, total_eval_losses = [], []
    total_train_accs, total_eval_accs = [], []
    best_acc = 0.0
    scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        print()
        print('epoch:{:d}/{:d}'.format(epoch, args.epochs))
        print('*' * 100)

        print("Training")
        epoch_train_loss, epoch_train_acc = train()
        epoch_train_loss = np.array(epoch_train_loss).mean()
        epoch_train_acc = np.array(epoch_train_acc).mean()
        print("Average loss:{:.4f}".format(epoch_train_loss))
        print("Average acc:{:.4f}".format(epoch_train_acc))
        total_train_losses.append(epoch_train_loss)
        total_train_accs.append(epoch_train_acc)
        print()

        print("Evaluating")
        epoch_eval_loss, epoch_eval_acc = evaluate()
        epoch_eval_loss = np.array(epoch_eval_loss).mean()
        epoch_eval_acc = np.array(epoch_eval_acc).mean()
        print("Average loss:{:.4f}".format(epoch_eval_loss))
        print("Average loss:{:.4f}".format(epoch_eval_acc))
        total_eval_losses.append(epoch_eval_loss)
        total_eval_accs.append(epoch_eval_acc)
        print()

        if epoch_eval_acc > best_acc:
            best_acc = epoch_eval_acc
            best_model = emojimodel
            torch.save(best_model, 'best_model.pt')

    x = range(args.epochs)
    plt.figure()
    plt.title('train and valid loss of double lstm')
    plt.xlabel(u'epochs')
    plt.ylabel(u'train and valid loss')
    plt.plot(x, total_train_losses, label='training loss')
    plt.plot(x, total_eval_losses, label='validation loss')
    plt.legend()
    plt.savefig('./train and valid loss curves of double lstm')

    plt.figure()
    plt.title('train and valid acc of double lstm')
    plt.xlabel(u'epochs')
    plt.ylabel(u'train and valid acc')
    plt.plot(x, total_train_accs, label='training acc')
    plt.scatter(total_train_accs.index(max(total_train_accs)), max(total_train_accs),
                label='best training acc = {:.4f}'.format(max(total_train_accs)))
    plt.plot(x, total_eval_accs, label='validation acc')
    plt.scatter(total_eval_accs.index(max(total_eval_accs)), max(total_eval_accs),
                label='best validation acc = {:.4f}'.format(max(total_eval_accs)))
    plt.legend()
    plt.savefig('./train and valid acc curves of double lstm')

# test top1/5 acc on test dataset
def Eval(model, sents, labels, print_every=1000):
    print("Evaluating the best model...")
    model.eval()

    n = sents.shape[0]
    y_real, y_pred = [], []
    top5_correct = 0
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    for idx, (sents, data_len, labels) in enumerate(dataloader):
        if (idx+1) % print_every == 0:
            print("{} / {}".format(idx+1, n))
        sents = sents.to(device)
        data_len = data_len.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(sents, data_len)

        probs = torch.nn.functional.softmax(outputs, dim=-1).cpu().numpy().squeeze()
        top1 = probs.argmax()
        top5 = np.argpartition(-probs, kth=5)[:5]
        y_pred.append(top1)
        y_real.append(labels[0])
        if label[0] in top5:
            top5_correct += 1

    print("Top5 Acc: {:.6f}".format(top5_correct/n))

# test single sentence
def emojize_novel_text(model, text):
    sents = []
    tokenized_text = text.split()
    for word in tokenized_text:
        if word in voc.keys():
            sents.append(voc[word])
        else:
            sents.append(voc['<UNK>'])
    print(sents, tokenized_text)
    tokens_tensor = torch.tensor([sents]).to(device)
    data_len = torch.tensor([len(sents)]).to(device)
    # tokens_tensor = tokens_tensor.to('cuda')
    # print(tokens_tensor)

    with torch.no_grad():
        outputs = model(tokens_tensor, data_len)

    probs = torch.nn.functional.softmax(outputs, dim=-1).cpu().numpy().squeeze()
    # print(probs)
    for idx in np.argpartition(-probs, kth=5)[:5]:
        print(emoji.emojize(id2label[idx]))

# model = torch.load('./best_model.pt').to(device)
# sents = np.load('./data/Xtest.npy', allow_pickle = True)
# labels = np.load('./data/ytest.npy', allow_pickle = True)
# Eval(model, sents, labels)
# emojize_novel_text(model, 'I love deep learning')
# emojize_novel_text(model, 'But I hate doing homework')
# emojize_novel_text(model, 'I would marry her on the spot')
# emojize_novel_text(model, 'nobody can say beyonce cant sing')
# emojize_novel_text(model, '60 Russians expelled from the US Now if we could just get the ONE out of the White House')