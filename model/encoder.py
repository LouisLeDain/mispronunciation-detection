import torch 
import torch.nn as nn
import torch.nn.functional as F

from sub import CNNStack, RNNStack

class AcousticEncoder(nn.Module):
    def __init__(self, input_size):
        super(AcousticEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = input_size // 2

        self.cnn_1 = CNNStack(input_channels=1, output_channels=1)
        self.cnn_2 = CNNStack(input_channels=1, output_channels=1)

        self.rnn_1 = RNNStack(input_size=self.input_size, hidden_size=self.hidden_size)
        self.rnn_2 = RNNStack(input_size=self.input_size, hidden_size=self.hidden_size)
        self.rnn_3 = RNNStack(input_size=self.input_size, hidden_size=self.hidden_size)
        self.rnn_4 = RNNStack(input_size=self.input_size, hidden_size=self.hidden_size)

    def forward(self, x):
        # x: (batch_size, channels, seq_length, input_size)
        x = self.cnn_1(x)
        x = self.cnn_2(x)

        # Reshape for RNN input: (batch_size, seq_length, input_size)
        x = x.squeeze(1)

        x = self.rnn_1(x)
        x = self.rnn_2(x)
        x = self.rnn_3(x)
        x = self.rnn_4(x)

        return x # (batch_size, seq_length, input_size)

class PhoneticEncoder(nn.Module):
    def __init__(self, input_size):
        super(PhoneticEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = input_size // 2

        self.cnn_1 = CNNStack(input_channels=1, output_channels=1)
        self.cnn_2 = CNNStack(input_channels=1, output_channels=1)

        self.rnn_1 = RNNStack(input_size=self.input_size, hidden_size=self.hidden_size)

    def forward(self, x):
        # x: (batch_size, channels, seq_length, input_size)
        x = self.cnn_1(x)
        x = self.cnn_2(x)

        # Reshape for RNN input: (batch_size, seq_length, input_size)
        x = x.squeeze(1)

        x = self.rnn_1(x)

        return x

class LinguisticEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(LinguisticEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirection

    def forward(self, x):
        # x shape: (batch_size, seq_length)
        # Embed the input
        embedded = self.embedding(x)  
        lstm_out, _ = self.bilstm(embedded)
        linear_out = self.linear(lstm_out)

        return lstm_out, linear_out
