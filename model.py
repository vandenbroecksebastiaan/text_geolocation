from torch import nn
from transformers import XLMRobertaModel, AutoModelForSequenceClassification
from config import config

class BaseRobertaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model = XLMRobertaModel \
            .from_pretrained("xlm-roberta-base",
                             num_labels=5, output_hidden_states=True)
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(98304, 6)
        self.bn1 = nn.BatchNorm1d(num_features=98304)

    def forward(self, input_ids, attention_mask):
        roberta_output = self.pretrained_model(input_ids, attention_mask)
        last_hidden_state = roberta_output.last_hidden_state
        last_hidden_state = last_hidden_state.flatten(start_dim=1, end_dim=2)
        last_hidden_state = self.relu(last_hidden_state)
        last_hidden_state = self.bn1(last_hidden_state)
        return self.fc1(last_hidden_state)


class BaseRobertaLanguageDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model = AutoModelForSequenceClassification \
            .from_pretrained("papluca/xlm-roberta-base-language-detection",
                             output_hidden_states=True)
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(98304, config["n_classes"])
        self.bn1 = nn.BatchNorm1d(num_features=98304)

    def forward(self, input_ids, attention_mask):
        roberta_output = self.pretrained_model(input_ids, attention_mask)
        last_hidden_state = roberta_output.hidden_states[-1]
        last_hidden_state = last_hidden_state.flatten(start_dim=1, end_dim=2)
        last_hidden_state = self.relu(last_hidden_state)
        last_hidden_state = self.bn1(last_hidden_state)
        return self.fc1(last_hidden_state)