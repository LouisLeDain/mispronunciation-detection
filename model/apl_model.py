import torch 
import torch.nn as nn
import torch.nn.functional as F

from encoder import AcousticEncoder, PhoneticEncoder, LinguisticEncoder
from decoder import Decoder

class APLModel(nn.Module):
    def __init__(self, input_size, vocab_size, embedding_dim, hidden_size, output_size, decoder_size):
        super(APLModel, self).__init__()
        self.acoustic_encoder = AcousticEncoder(input_size=input_size)
        self.phonetic_encoder = PhoneticEncoder(input_size=input_size)
        self.linguistic_encoder = LinguisticEncoder(vocab_size, embedding_dim, hidden_size, output_size)
        self.decoder = Decoder(input_size=decoder_size , output_size=vocab_size)

    def forward(self, acoustic_input, phonetic_input, phoneme_input):
        channel_acoustic_input = acoustic_input.unsqueeze(1)
        channel_phonetic_input = phonetic_input.unsqueeze(1)

        acoustic_encoding = self.acoustic_encoder(channel_acoustic_input)
        phonetic_encoding = self.phonetic_encoder(channel_phonetic_input)

        query = torch.cat((acoustic_encoding, phonetic_encoding), dim=-1)  # Concatenate encodings
        values, keys = self.linguistic_encoder(phoneme_input)

        logits = self.decoder(query, keys, values)

        return logits # (batch_size, seq_length, vocab_size)