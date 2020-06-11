import numpy as np
import codecs
import pickle
import random
import torch
import emoji
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence


def preprocess(data_path, split_ratio):
    with codecs.open(data_path, encoding='utf-8') as f:
        content = f.read().split('\n\n')

    # initialize
    word2id, id2word, label2id, id2label = {}, {}, {}, {}
    sentence_ids, labels = [], []
    word2id['<PAD>'] = 0
    id2word[0] = '<PAD>'
    word2id['<UNK>'] = 1
    id2word[1] = '<UNK>'
    word_count = 2
    label_count = 0

    print("Start preprocessing...")
    for idx, line in enumerate(content):
        pairs = line.split('\n')[1:-1]
        sentence_id, label = [], []
        for pair in pairs:
            try:
                word, tag = pair.split()
            except:
                print("format error for {}".format(idx))
                continue
            if word not in word2id.keys():
                word2id[word] = word_count
                id2word[word_count] = word
                word_count += 1
            if tag != 'O' and tag not in label2id.keys():
                label2id[tag] = label_count
                id2label[label_count] = tag
                label_count += 1
            sentence_id.append(word2id[word])
            if tag != 'O':
                label.append(label2id[tag])
        sentence_ids.append(sentence_id)
        labels.append(label)
    print("Preprocessing finished.")

    print("There are {} types of emojis.".format(label_count))
    print("There are {} tokens in total.".format(word_count))
    print("There are {} sentences before filtering.".format(len(sentence_ids)))

    filtered_sentences, filtered_labels = [], []
    for idx, label in enumerate(labels):
        if len(label) > 0:
            filtered_sentences.append(sentence_ids[idx])
            filtered_labels.append(list(set(labels[idx])))
    print(len(filtered_sentences), len(filtered_labels))
    print("There are {} sentences after filtering.".format(len(filtered_sentences)))

    with open("./data/word2id.pickle", "wb") as f:
        pickle.dump(word2id, f)

    with open("./data/id2word.pickle", "wb") as f:
        pickle.dump(id2word, f)

    with open("./data/label2id.pickle", "wb") as f:
        pickle.dump(label2id, f)

    with open("./data/id2label.pickle", "wb") as f:
        pickle.dump(id2label, f)

    X_train, X_test, y_train, y_test = train_test_split(filtered_sentences, filtered_labels, test_size=split_ratio, shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=split_ratio, shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=split_ratio, shuffle=False)
    np.save('./data/Xtrain', X_train)
    np.save('./data/ytrain', y_train)
    np.save('./data/Xtest', X_test)
    np.save('./data/ytest', y_test)


def collate_fn(batch_data):
    batch_data.sort(key=lambda xi: len(xi[0]), reverse=True)
    data_length = [len(xi[0]) for xi in batch_data]
    sent_seq = [xi[0] for xi in batch_data]
    label = [xi[1] for xi in batch_data]
    padded_sent_seq = pad_sequence(sent_seq, batch_first=True, padding_value=0)
    return padded_sent_seq, torch.tensor(data_length, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def plotdata(sents, labels):
    # print(labels[:10])
    with open("./data/id2label.pickle", "rb+") as f:
        id2label = pickle.load(f)
    sents_len,label_counts = {}, {}
    for sent in sents:
        if len(sent) in sents_len.keys():
            sents_len[len(sent)] += 1
        else:
            sents_len[len(sent)] = 1
    for label in labels:
        for lab in label:
            if lab in label_counts.keys():
                label_counts[lab] += 1
            else:
                label_counts[lab] = 1
    sorted_sents_len, sorted_label_counts = {}, {}
    for idx in sorted(label_counts):
        sorted_label_counts[idx] = label_counts[idx]
    for idx in sorted(sents_len):
        sorted_sents_len[idx] = sents_len[idx]

    plt.figure(figsize=(20, 8))
    plt.title('lengths of the sentences', fontsize=30)
    plt.bar(range(len(sorted_sents_len)), sorted_sents_len.values())
    plt.xticks(range(len(sorted_sents_len)), sorted_sents_len.keys(), fontsize=16)
    plt.savefig('./sentence length')

    plt.figure(figsize=(20, 8))
    plt.title('numbers of the emojis', fontsize=30)
    plt.bar(range(len(sorted_label_counts)), sorted_label_counts.values())
    plt.xticks(range(len(sorted_label_counts)), sorted_label_counts.keys(), fontsize=16)
    plt.savefig('./numbers of the emojis')

    top_emoji = sorted(label_counts.items(), key = lambda kv:(kv[1], kv[0]))[-5:]
    for emoji_pair in top_emoji:
        print(emoji.emojize(id2label[emoji_pair[0]]))


class EmojiDataset(Dataset):
    def __init__(self, x_data_path, y_data_path):
        self.X_data = np.load(x_data_path, allow_pickle = True)
        self.y_data = np.load(y_data_path, allow_pickle = True)

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        sentence = self.X_data[idx]
        label = random.choice(self.y_data[idx])
        return torch.tensor(sentence), label