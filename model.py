from torch import nn
import torch
from transformers import AutoModel, XLMRobertaForSequenceClassification, XLMRobertaModel


class RobertaModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.pretrained_model = XLMRobertaModel.from_pretrained("xlm-roberta-base", num_labels=5, output_hidden_states=True)


        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(98304,
                             6)
        
        self.bn1 = nn.BatchNorm1d(num_features=98304)
        self.sm = nn.Softmax()

    def forward(self, input_ids, attention_mask):
        roberta_output = self.pretrained_model(input_ids, attention_mask)
        last_hidden_state = roberta_output.last_hidden_state
        last_hidden_state = last_hidden_state.flatten(start_dim=1, end_dim=2)

        last_hidden_state = self.relu(last_hidden_state)
        last_hidden_state = self.bn1(last_hidden_state)
        return self.fc1(last_hidden_state)

