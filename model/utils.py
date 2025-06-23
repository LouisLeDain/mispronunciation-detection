import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import librosa
from transformers import AutoProcessor, AutoModelForCTC

PHONEMES = ['AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AY', 'B', 'CH', 'D', 
            'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 
            'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 
            'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH',
            '[PAD]']

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
        
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, keys, values):
        scores = torch.bmm(query, keys.transpose(1, 2))
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, values)
        return context    

def beam_search_over_frames(Y, beam_width=3, sequence_length=10):
    """
    Perform beam search over frame-wise probability sequences for a batch of inputs.

    :param Y: A numpy array of shape [B, T, S] containing frame-wise probabilities for a batch.
    :param beam_width: The number of sequences to keep track of at each step.
    :param sequence_length: The desired length of the output sequence.
    :return: A list of the most likely sequences found by beam search for each batch element.
    """
    B, T, S = Y.shape
    batch_best_sequences = []

    for b in range(B):
        beams = [([], 0.0)]

        for t in range(T):
            current_probs = Y[b, t]
            new_beams = []

            for sequence, log_prob in beams:
                if len(sequence) == sequence_length:
                    # If the sequence has reached the desired length, carry it over without changes
                    new_beams.append((sequence, log_prob))
                    continue

                # Calculate log probabilities for each possible next token
                log_probs = np.log(current_probs)

                # Find the top `beam_width` candidates
                top_candidates = np.argpartition(log_probs, -beam_width)[-beam_width:]

                for candidate in top_candidates:
                    new_sequence = sequence + [candidate]
                    new_log_prob = log_prob + log_probs[candidate]
                    new_beams.append((new_sequence, new_log_prob))

            # Sort all new beams by their log probability
            new_beams.sort(key=lambda x: x[1], reverse=True)

            # Select the top `beam_width` beams for the next iteration
            beams = new_beams[:beam_width]

        # Return the best beam for the current batch element
        best_sequence, _ = max(beams, key=lambda x: x[1])
        batch_best_sequences.append(best_sequence)
    # Convert sequences to torch tensors for consistency and concatenate sequences
    batch_best_sequences_list = [torch.tensor(seq) for seq in batch_best_sequences]
    batch_best_sequences = torch.stack(batch_best_sequences_list, dim=0)

    return batch_best_sequences

class Tokenizer:
    def __init__(self, phonemes=PHONEMES):
        self.phonemes = phonemes
        self.phoneme_to_id = {phoneme: idx for idx, phoneme in enumerate(phonemes)}
        self.id_to_phoneme = {idx: phoneme for idx, phoneme in enumerate(phonemes)}

    def split(self, text):
        return text.split(sep=' ')

    def encode(self, sequence):
        return [self.phoneme_to_id[phoneme] for phoneme in sequence]

    def decode(self, ids):
        return [self.id_to_phoneme[idx] for idx in ids]

def capture_transformer_network_outputs(audio_file):
    processor = AutoProcessor.from_pretrained("mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme")
    model = AutoModelForCTC.from_pretrained("mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme")

    audio_input, sample_rate = librosa.load(audio_file, sr=16000)
    input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values

    layer_transformer_outputs = []

    def transformer_hook(module, input, output):
        layer_transformer_outputs.append(output)

    for layer in model.wav2vec2.encoder.layers:
        layer.register_forward_hook(transformer_hook)
    
    with torch.no_grad():
        outputs = model(input_values)
    
    for layer in model.wav2vec2.encoder.layers:
        layer._forward_hooks.clear()
    
    return layer_transformer_outputs

