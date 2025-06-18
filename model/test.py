import torch
from sub import CNNStack, RNNStack
from encoder import AcousticEncoder, PhoneticEncoder, LinguisticEncoder


## Testing CNNStack
print("Testing CNNStack...")

cnn_stack = CNNStack(input_channels=1, output_channels=1)
input_tensor = torch.randn(1, 1, 81, 64)  # (batch_size, channels, height, width)
output_tensor = cnn_stack(input_tensor)
print(f"CNNStack output shape: {output_tensor.shape}")  # Expected: (1, 1, 81, 64)

## Testing RNNStack
print("Testing RNNStack...")

rnn_stack = RNNStack(input_size=64, hidden_size=32)
input_tensor_rnn = torch.randn(1, 81, 64)  # (batch_size, seq_length, input_size)
output_tensor_rnn = rnn_stack(input_tensor_rnn)
print(f"RNNStack output shape: {output_tensor_rnn.shape}")  # Expected: (1, 81, 64)

## Testing AcousticEncoder
print("Testing AcousticEncoder...")

acoustic_encoder = AcousticEncoder(input_size=64)
input_tensor_encoder = torch.randn(1, 1, 81, 64)
output_tensor_encoder = acoustic_encoder(input_tensor_encoder)
print(f"AcousticEncoder output shape: {output_tensor_encoder.shape}")  # Expected: (1, 81, 64)

## Testing PhoneticEncoder
print("Testing PhoneticEncoder...")

phonetic_encoder = PhoneticEncoder(input_size=64)
input_tensor_phonetic = torch.randn(1, 1, 81, 64)
output_tensor_phonetic = phonetic_encoder(input_tensor_phonetic)
print(f"PhoneticEncoder output shape: {output_tensor_phonetic.shape}")

## Testing LinguisticEncoder
print("Testing LinguisticEncoder...")

linguistic_encoder = LinguisticEncoder(vocab_size=40, embedding_dim=64, hidden_size=32, output_size=64)
input_tensor_linguistic = torch.randint(0, 40, (1, 10))  # (batch_size, seq_length)
v_output, k_output = linguistic_encoder(input_tensor_linguistic)
print(f"LinguisticEncoder v_output shape: {v_output.shape}")  # Expected: (1, 10, 64)
print(f"LinguisticEncoder k_output shape: {k_output.shape}")  # Expected:

