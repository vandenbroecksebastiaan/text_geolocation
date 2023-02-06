from torch import nn
import torch


class Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=1024,
                            num_layers=1, batch_first=False, bidirectional=False)

        self.fc1 = nn.Linear(1024, 1024*2)
        self.fc2 = nn.Linear(1024*2, 1024)
        self.fc3 = nn.Linear(1024, 2)

        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(num_features=1024*2)
        self.batchnorm2 = nn.BatchNorm1d(num_features=1024)
        self.dropout = nn.Dropout(0.8)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = out[-1]

        out = self.fc1(out)
        out = self.batchnorm1(out)
        out = self.relu(out)
        # out = self.dropout(out)

        out = self.fc2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        # out = self.dropout(out)

        return self.fc3(out)