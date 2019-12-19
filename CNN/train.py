# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 10:48:20 2019

@author: huxiao
"""

import jieba
from torchtext import data
import re

# from torchtext.vocab import Vectors
# from torchtext.vocab import GloVe

BATCH_SIZE = 128
EMBEDDING_DIM = 128
FIX_LEN = 100
VOCAB_SIZE = 50002
LABEL_NUM = 14
DATASET_PATH = './'


def tokenizer(text):
    r = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')
    text = r.sub(' ', text)
    return [word for word in jieba.cut(text) if word.strip()]


# 去停用词
def get_stop_words():
    with open('stopwords.txt', encoding="utf-8") as f:
        stop_words = []
        for line in f.readlines():
            line = line[:-1]
            line = line.strip()
            stop_words.append(line)
    return stop_words


def load_data():
    print('data loading...')
    stop_words = get_stop_words()

    Text = data.Field(sequential=True, lower=True, tokenize=tokenizer, fix_length=FIX_LEN, stop_words=stop_words,
                      batch_first=True)
    Label = data.LabelField(sequential=False, use_vocab=False)
    Text.tokenize = tokenizer

    print("datasets processing...")
    train, val = data.TabularDataset.splits(
        path=DATASET_PATH,
        skip_header=True,
        train='compress_train_num.tsv',
        validation='compress_val_num.tsv',
        # test='compress_test_num.tsv',
        format='tsv',
        fields=[('index', None), ('label', Label), ('text', Text)],
    )

    print("vocab building...")
    Text.build_vocab(train, val, max_size=VOCAB_SIZE)
    Label.build_vocab(train, val)

    print("batch spliting...")
    train_iter, val_iter = data.Iterator.splits(
        (train, val),
        sort_key=lambda x: len(x.text),
        batch_sizes=(BATCH_SIZE, len(val)),
        device="cpu"
    )
    # print(len(Text.vocab))
    # print(len(Label.vocab))

    print("data loaded!!!")
    return train_iter, val_iter, len(Text.vocab), len(Label.vocab), Text, Label


# Model
FILTER_NUM = 100
FILTER_SIZES = [3, 4, 5]
DROPOUT = 0.5
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):

    def __init__(self):
        super(TextCNN, self).__init__()

        label_num = LABEL_NUM
        filter_num = FILTER_NUM
        filter_sizes = FILTER_SIZES

        vocab_size = VOCAB_SIZE
        embedding_dim = EMBEDDING_DIM

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, filter_num, (fs, embedding_dim)) for fs in filter_sizes])
        self.dropout = nn.Dropout(DROPOUT)
        self.linear = nn.Linear(len(filter_sizes) * filter_num, label_num)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), 1, x.size(1), EMBEDDING_DIM)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]
        x = [x_item.view(x_item.size(0), -1) for x_item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        y = self.linear(x)
        return y


# train
import sys
import os

EPOCH_SIZE = 10
LR = 0.001
VAL_INTERVAL = 100
SAVE_BEST = True
SAVE_DIR = 'model_vocab_dir'
EARLY_STOPPING = 500

train_history = {'acc': [], 'loss': []}
train_evaluation = {'y_true': [], 'y_pred': []}

val_history = {'acc': [], 'loss': []}
val_evaluation = {'y_true': [], 'y_pred': []}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(train_iter, val_iter):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    steps = 0
    best_acc = 0
    last_step = 0
    for epoch in range(1, EPOCH_SIZE + 1):
        print("\nEpoch %d\n" % (epoch))
        for batch in train_iter:
            x, y_true = batch.text, batch.label
            x, y_true = x.to(device), y_true.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = F.cross_entropy(y_pred, y_true)
            loss.backward()
            optimizer.step()
            steps += 1
            corrects = (torch.max(y_pred, 1)[1].view(y_true.size()).data == y_true.data).sum()
            train_acc = 100.0 * corrects / batch.batch_size
            train_history['acc'].append(train_acc / 100)
            train_history['loss'].append(loss.item())
            train_evaluation['y_pred'].extend(torch.max(y_pred, 1)[1].data.tolist())
            train_evaluation['y_true'].extend(y_true.data.tolist())
            sys.stdout.write('\rBatch[%d] - loss: %.6f  acc: %.4f %%(%d/%d)'
                             % (steps, loss.item(), train_acc, corrects, batch.batch_size))
            if steps % VAL_INTERVAL == 0:
                dev_acc = evaluate(val_iter, model)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if SAVE_BEST:
                        print('Saving best model, acc: %.4f%%\n' % (best_acc))
                        save(model, SAVE_DIR, 'best', steps)
                else:
                    if steps - last_step >= EARLY_STOPPING:
                        print('\nearly stop by %d steps, acc: %.4f%%'.format(steps, best_acc))
                        return -1


def evaluate(val_iter, model, stat=False):
    model.eval()

    for batch in val_iter:
        x, y_true = batch.text, batch.label
        x, y_true = x.to(device), y_true.to(device)
        y_pred = model(x)
        loss = F.cross_entropy(y_pred, y_true)
        val_loss = loss.item()
        corrects = (torch.max(y_pred, 1)[1].view(y_true.size()) == y_true).sum()
        if stat:
            val_history['acc'].append(corrects / batch.batch_size)
            val_history['loss'].append(loss.item())
            val_evaluation['y_pred'].extend(torch.max(y_pred, 1)[1].data.tolist())
            val_evaluation['y_true'].extend(y_true.data.tolist())

    size = len(val_iter.dataset)
    val_acc = 100.0 * corrects / size
    print('\nEvaluation - loss: %.6f  acc: %.4f%%(%d/%d) \n'
          % (val_loss, val_acc, corrects, size))
    return val_acc


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '%s_steps_%d.pt' % (save_prefix, steps)
    torch.save({"model": model, "vocab": VOCAB_TEXT.vocab}, save_path)


# evaluation
CLASSES = ["体育", "娱乐", "家居", "彩票", "房产", "教育", "时尚", "时政", "星座", "游戏", "社会", "科技", "股票", "财经"]

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font="simhei")

import numpy as np
import pandas as pd


def show_history(history):
    plt.figure(figsize=(6, 6))
    plt.plot(history['acc'], color='green')
    plt.plot(history['loss'], color='red')
    plt.title('History')
    plt.ylabel('acc/loss')
    plt.xlabel('Batch')
    plt.legend(['acc', 'loss'], loc='upper right')
    plt.show()


def show_confusion_matrix(evaluation):
    f, ax = plt.subplots(figsize=(10, 10))
    df = pd.DataFrame(confusion_matrix(evaluation['y_true'], evaluation['y_pred']), columns=CLASSES, index=CLASSES)
    # print(df)
    ax = sns.heatmap(df, annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5)
    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=360, horizontalalignment='right', fontsize=13)
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=45, horizontalalignment='right', fontsize=13)
    #f.savefig("confusion_matrix.png", bbox_inches='tight', dpi=200)
    plt.show()


def show_statistics(evaluation):
    p, r, f1, s = precision_recall_fscore_support(evaluation['y_true'], evaluation['y_pred'])
    avg_p = np.average(p, weights=s)
    avg_r = np.average(r, weights=s)
    avg_f1 = np.average(f1, weights=s)
    total_s = np.sum(s)
    df1 = pd.DataFrame({'类别': CLASSES, 'Precision': p, 'Recall': r, 'F1': f1, 'Support': s})
    df2 = pd.DataFrame(
        {'类别': ['全部'], 'Precision': [avg_p], 'Recall': [avg_r], 'F1': [avg_f1], 'Support': [total_s]})
    df2.index = [14]
    df = pd.concat([df1, df2])


def show():
    show_history(train_history)
    # show_history(val_history)
    show_confusion_matrix(val_evaluation)
    show_statistics(val_evaluation)


if __name__ == "__main__":
    # preprocess data
    train_iter, val_iter, VOCAB_SIZE, LABEL_NUM, VOCAB_TEXT, VOCAB_LABEL = load_data()
    # model
    model = TextCNN()
    # train
    train(train_iter, val_iter)
    # evaluation
    evaluate(val_iter, model, stat=True)
    show()
