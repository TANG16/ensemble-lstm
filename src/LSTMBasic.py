import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMBasic(nn.ModuleList):
    def __init__(self, args):
        super(LSTMBasic, self).__init__()

        # Hyperparameters
        self.batch_size = args.batch_size
        self.hidden_dim = args.hidden_dim
        self.LSTM_layers = args.lstm_layers
        self.input_size = args.max_words

        self.dropout = nn.Dropout(0.5)
        # self.embedding = nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=6,
            hidden_size=self.hidden_dim,
            num_layers=self.LSTM_layers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(
            in_features=self.hidden_dim, out_features=self.hidden_dim * 2
        )
        self.fc2 = nn.Linear(self.hidden_dim * 2, 1)

    def forward(self, x):

        # Hidden and cell state definion
        h = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim))
        c = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim))

        # Initialization fo hidden and cell states
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)

        # Each sequence "x" is passed through an embedding layer
        out = x
        # print(out.shape)
        # print(h.shape)
        # print(c.shape)

        # Feed LSTMs
        # out, (hidden, cell) = self.lstm(out, (h, c))
        out, (hidden, cell) = self.lstm(out, (h, c))

        out = self.dropout(out)
        # The last hidden state is taken
        out = torch.relu_(self.fc1(out[:, -1, :]))
        out = self.dropout(out)
        out = torch.sigmoid(self.fc2(out))

        return out
