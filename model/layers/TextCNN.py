import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule

class TextCNN(nn.Module):
    def __init__(self, word_embedding_dimension, sentence_max_size, label_num):
        super(TextCNN, self).__init__()
        Dim = word_embedding_dimension
        Cla = label_num
        Ci = 1
        Knum = 2
        Ks = [2, 3, 4]
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Knum, (K, Dim)) for K in Ks])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(Ks) * Knum, Cla)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.fc(x)
        return logit
