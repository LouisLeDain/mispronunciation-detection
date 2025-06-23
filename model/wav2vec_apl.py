from apl_model import APLModel
from input import AcousticInput, PhoneticInput
from utils import capture_transformer_network_outputs, Tokenizer

import torch 
import torch.nn as nn
import torch.nn.functional as F

class Wav2VecAPL(nn.Module):
    def __init__(self, input_size, vocab_size=41, embedding_dim, hidden_size, output_size, decoder_size, n_inputs = 24):
        super(Wav2VecAPL, self).__init__()
        self.acoustic_input = AcousticInput(n_inputs)
        self.phonetic_input = PhoneticInput(n_inputs)
        self.apl_model = APLModel(input_size, vocab_size, embedding_dim, hidden_size, output_size, decoder_size)
        self.tokenizer = Tokenizer()

    def forward(self, audio_files, phonemes):
        # Process audio files through transformer network
        batched_audio = torch.stack([capture_transformer_network_outputs(audio_file) for audio_file in audio_files], dim = 0) # (batch_size, n_inputs, seq_length, input_size)

        # Compute acoustic and phonetic inputs
        acoustic_input = self.acoustic_input(batched_audio)  # (batch_size, seq_length, input_size)
        phonetic_input = self.phonetic_input(batched_audio)  # (batch_size, seq_length, input_size)

        # Convert phonemes to tensor
        phonemic_input = torch.stack([self.tokenizer.phonemes_to_tensor(phoneme) for phoneme in phonemes], dim=0)

        # Forward pass through APL model
        output_logits = self.apl_model(acoustic_input, phonetic_input, phonemes)  # (batch_size, seq_length, vocab_size)

        return output_logits

        