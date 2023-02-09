from torch import nn
import torch
from transformers import AutoModel

class Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=1024,
                            num_layers=2, batch_first=False, bidirectional=True)

        self.fc1 = nn.Linear(1024*2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)

        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(num_features=512)
        self.batchnorm2 = nn.BatchNorm1d(num_features=256)
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



class RobertaModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.pretrained_model = AutoModel.from_pretrained("xlm-roberta-large",
                                                          num_labels=6)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.pretrained_model.config.hidden_size,
                             self.pretrained_model.config.hidden_size)
        self.fc2 = nn.Linear(self.pretrained_model.config.hidden_size,
                             6)
        self.sm = nn.Softmax(dim=1)
        
    def forward(self, input_ids, attention_mask):
        roberta_output = self.pretrained_model(input_ids, attention_mask)
        hidden_state = torch.mean(roberta_output.last_hidden_state, 1)
        
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.relu(hidden_state)
        hidden_state = self.fc2(hidden_state)
        
        return self.sm(hidden_state)
