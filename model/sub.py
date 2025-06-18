import torch 
import torch.nn as nn
import torch.nn.functional as F

class CNNStack(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNNStack, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x): #(batch_size, channels, height, width)
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x #(batch_size, output_channels, height, width)

class RNNStack(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNStack, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)  # For bidirectional, multiply by 2
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x): #(batch_size, seq_length, input_size)
        x, _ = self.rnn(x)
        x = self.batch_norm(x.permute(0, 2, 1)).permute(0, 2, 1) 
        x = self.dropout(x)
        return x # (batch_size, seq_length, hidden_size * 2)
        
