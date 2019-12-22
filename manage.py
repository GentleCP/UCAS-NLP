#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from NaiveBayes.naive_bayes import NaiveBayes

# Model
BATCH_SIZE = 128
EMBEDDING_DIM = 128
FIX_LEN = 100
VOCAB_SIZE = 50002
LABEL_NUM = 14
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


def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Website.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
