import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule


# class TextCNN(BasicModule):
#     def __init__(self, word_embedding_dimension, sentence_max_size, label_num):
#         super(TextCNN, self).__init__()
#         self.label_num = label_num
#         self.conv3 = nn.Conv2d(1, 1, (3, word_embedding_dimension))
#         self.conv4 = nn.Conv2d(1, 1, (4, word_embedding_dimension))
#         self.conv5 = nn.Conv2d(1, 1, (5, word_embedding_dimension))
#         self.Max3_pool = nn.MaxPool2d((sentence_max_size-3+1, 1))
#         self.Max4_pool = nn.MaxPool2d((sentence_max_size-4+1, 1))
#         self.Max5_pool = nn.MaxPool2d((sentence_max_size-5+1, 1))
#         self.linear1 = nn.Linear(3, label_num)
#
#     def forward(self, x):
#         batch = x.shape[0]
#         # Convolution
#         x1 = F.relu(self.conv3(x))
#         x2 = F.relu(self.conv4(x))
#         x3 = F.relu(self.conv5(x))
#
#         # Pooling
#         x1 = self.Max3_pool(x1)
#         x2 = self.Max4_pool(x2)
#         x3 = self.Max5_pool(x3)
#
#         # capture and concatenate the features
#         x = torch.cat((x1, x2, x3), -1)
#         x = x.view(batch, 1, -1)
#
#         # project the features to the labels
#         x = self.linear1(x)
#         x = x.view(-1, self.label_num)
#
#         return x


class TextCNN(nn.Module):
    def __init__(self, word_embedding_dimension, sentence_max_size, label_num):
        super(TextCNN, self).__init__()
        Dim = word_embedding_dimension
        Cla = label_num
        Ci = 1
        Knum = 2
        Ks = [2, 3, 4]
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Knum, (K, Dim)) for K in Ks])  ## 卷积层
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(Ks) * Knum, Cla)  ##全连接层

    def forward(self, x):
        x = self.embed(x)  # (N,W,D)

        x = x.unsqueeze(1)  # (N,Ci,W,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # len(Ks)*(N,Knum,W)
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]  # len(Ks)*(N,Knum)

        x = torch.cat(x, 1)  # (N,Knum*len(Ks))

        x = self.dropout(x)
        logit = self.fc(x)
        return logit
