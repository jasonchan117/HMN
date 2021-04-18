import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule


class TextCNN(BasicModule):
    def __init__(self, word_embedding_dimension, sentence_max_size, label_num):
        super(TextCNN, self).__init__()
        self.label_num = label_num
        self.conv3 = nn.Conv2d(1, 1, (3, word_embedding_dimension))
        self.conv4 = nn.Conv2d(1, 1, (4, word_embedding_dimension))
        self.conv5 = nn.Conv2d(1, 1, (5, word_embedding_dimension))
        self.Max3_pool = nn.MaxPool2d((sentence_max_size-3+1, 1))
        self.Max4_pool = nn.MaxPool2d((sentence_max_size-4+1, 1))
        self.Max5_pool = nn.MaxPool2d((sentence_max_size-5+1, 1))
        self.linear1 = nn.Linear(3, label_num)

    def forward(self, x):
        batch = x.shape[0]
        # Convolution
        x1 = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x))
        x3 = F.relu(self.conv5(x))

        # Pooling
        x1 = self.Max3_pool(x1)
        x2 = self.Max4_pool(x2)
        x3 = self.Max5_pool(x3)

        # capture and concatenate the features
        x = torch.cat((x1, x2, x3), -1)
        x = x.view(batch, 1, -1)

        # project the features to the labels
        x = self.linear1(x)
        x = x.view(-1, self.label_num)

        return x
