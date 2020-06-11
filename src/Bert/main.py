from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from model import BertClassifier
from data import EmojiDataset, DataLoader, collate_fn
import torch
import numpy as np
import torch.nn as nn

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'


def train(model, device, train_loader, optimizer, scheduler, epoch, print_every=100):
    print("Training at epoch {}".format(epoch))
    est_batch = len(train_loader.dataset) / train_loader.batch_size

    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = []
    total_count, total_correct = 0, 0

    for idx, (sents, labels) in enumerate(train_loader):
        sents, labels = sents.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(sents)

        res = preds.argmax(axis=1).cpu().numpy()
        gt = labels.cpu().numpy()
        correct_count = (res==gt).sum()
        acc = correct_count / gt.shape[0]

        total_correct += correct_count
        total_count += gt.shape[0]
        total_loss.append(criterion(preds, labels).item())

        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        # optimizer.module.step()
        scheduler.step()

        if (idx+1) % print_every == 0:
            print("Batch: {} / {:.0f}, Loss: {:.6f}, Acc: {:.6f}".format(idx, est_batch, loss.item(), acc))

    loss, acc = np.array(total_loss).mean(), total_correct/total_count
    print("Average Loss: {:.6f}, Accuracy: {:.6f}".format(loss, acc))
    return loss, acc


def eval(model, device, test_loader):
    print("Evaluating...")
    est_batch = len(test_loader.dataset) / test_loader.batch_size

    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = []
    total_count, total_correct = 0, 0
    for idx, (sents, labels) in enumerate(test_loader):
        sents, labels = sents.to(device), labels.to(device)
        with torch.no_grad():
            preds = model(sents)

        res = preds.argmax(axis=1).cpu().numpy()
        gt = labels.cpu().numpy()
        total_correct += (res==gt).sum()
        total_count += gt.shape[0]
        total_loss.append(criterion(preds, labels).item())

    loss, acc = np.array(total_loss).mean(), total_correct/total_count
    print("Average Loss: {:.6f}, Accuracy: {:.6f}".format(loss, acc))
    return loss, acc


device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 30
best_acc = 0.0
eval_losses, eval_accs = [], []
train_losses, train_accs = [], []

model = BertClassifier(freeze_bert=False)
model = model.to(device)
# model = nn.DataParallel(model)

train_dataset = EmojiDataset('../../data/train_bert_sentences.npy', '../../data/train_bert_labels.npy')
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

test_dataset = EmojiDataset('../../data/test_bert_sentences.npy', '../../data/test_bert_labels.npy')
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
# optimizer = nn.DataParallel(optimizer)

total_steps = len(train_dataloader) * epochs
# scheduler = get_linear_schedule_with_warmup(
#   optimizer,
#   num_warmup_steps = 0,
#   num_training_steps = total_steps
# )

scheduler = get_constant_schedule_with_warmup(
  optimizer,
  num_warmup_steps = 1000,
)

for epoch in range(epochs):
    train_loss, train_acc = train(model, device, train_dataloader, optimizer, scheduler, epoch, print_every=50)
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    eval_loss, eval_acc = eval(model, device, test_dataloader)
    eval_losses.append(eval_loss)
    eval_accs.append(eval_acc)

    if (best_acc < eval_acc):
        best_acc = eval_acc
        torch.save(model, '../../model/bertclassifier_epoch_{}'.format(epoch))
        print("Model saved.")

    np.save('../../model/eval_losses', eval_losses)
    np.save('../../model/eval_accs', eval_accs)
    np.save('../../model/train_losses', train_losses)
    np.save('../../model/train_accs', train_accs)
