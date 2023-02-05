from torch import nn


class Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=1028,
                            num_layers=3, batch_first=False, bidirectional=True)

        self.fc1 = nn.Linear(2056, 1028)
        self.fc2 = nn.Linear(1028, 512)
        self.fc3 = nn.Linear(512, 2)

        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(num_features=1028)
        self.batchnorm2 = nn.BatchNorm1d(num_features=512)
        self.dropout = nn.Dropout(0.8)

    def forward(self, x):
        out, hn = self.lstm(x)
        out = out[-1, :, :]

        out = self.fc1(out)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.dropout(out)

        return self.fc3(out)