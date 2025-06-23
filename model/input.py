import torch 
import torch.nn as nn
import torch.nn.functional as F

from utils import capture_transformer_network_outputs

class AcousticInput(nn.Module):
    def __init__(self, n):
        super(AcousticInput, self).__init__()
        self.input_size = n
        self.acoustic_weights = nn.Parameter(torch.randn(n), requires_grad=True)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, inputs): # input of shape (batch, n, height, width)
        weights_normalized = self.softmax(self.acoustic_weights)  # Normalize weights using softmax

        reshaped_weights = weights_normalized.view(1, self.input_size, 1, 1)  # Reshape to (1, n, 1, 1)
        weighted_inputs = (inputs * reshaped_weights).sum(dim=1)  # Weighted sum across the n elements

        return weighted_inputs  # (batch_size, seq_length, hidden_size)

class PhoneticInput(nn.Module):
    def __init__(self, n):
        super(PhoneticInput, self).__init__()
        self.input_size = n
        self.phonetic_weights = nn.Parameter(torch.randn(n), requires_grad=True)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, inputs): # input of shape (batch, n, height, width)
        weights_normalized = self.softmax(self.phonetic_weights)  # Normalize weights using softmax

        reshaped_weights = weights_normalized.view(1, self.input_size, 1, 1)  # Reshape to (1, n, 1, 1)
        weighted_inputs = (inputs * reshaped_weights).sum(dim=1)  # Weighted sum across the n elements

        return weighted_inputs  # (batch_size, seq_length, hidden_size)