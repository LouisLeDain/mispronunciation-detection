import torch 
import torch.nn as nn
import torch.nn.functional as F

from utils import Attention

class Decoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Decoder, self).__init__()
        self.Attention = Attention()
        self.fc = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, keys, values):
        context = self.Attention(query, keys, values)
        concatenated = torch.cat((query, context), dim=-1)
        output = self.fc(concatenated)
        output = self.softmax(output)
        return output  # (batch_size, seq_length, output_size)

        