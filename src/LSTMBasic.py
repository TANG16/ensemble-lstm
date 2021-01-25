import torch
import torch.nn as nn
import torch.nn.functional as F
from config import input_size


class LSTMBasic(nn.ModuleList):
    def __init__(self, args):
        super(LSTMBasic, self).__init__()
        self.batch_size = args.batch_size
        self.hidden_dim = args.hidden_dim
        self.LSTM_layers = args.lstm_layers
        self.input_size = input_size
        self.dropout = nn.Dropout(0.5)  # not used ATM
        self.lstm = nn.LSTM(
            input_size=self.input_size,  # calculate this
            hidden_size=self.hidden_dim,
            num_layers=self.LSTM_layers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(
            in_features=self.hidden_dim, out_features=2
        )  # fully connected
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        out, (h, c) = self.lstm(x, None)
        out = self.fc1(out[:, -1, :])  # take last output
        out = self.softmax(out)
        return out[:, 1].unsqueeze(1)
