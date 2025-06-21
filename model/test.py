import torch
from utils import CNNStack, RNNStack, Attention, Tokenizer
from encoder import AcousticEncoder, PhoneticEncoder, LinguisticEncoder
from decoder import Decoder
from main import APLModel

## Testing CNNStack
print("\n Testing CNNStack...")

cnn_stack = CNNStack(input_channels=1, output_channels=1)
input_tensor = torch.randn(1, 1, 81, 64)  # (batch_size, channels, height, width)
output_tensor = cnn_stack(input_tensor)
print(f"CNNStack output shape: {output_tensor.shape}")  # Expected: (1, 1, 81, 64)

## Testing RNNStack
print("\n Testing RNNStack...")

rnn_stack = RNNStack(input_size=64, hidden_size=32)
input_tensor_rnn = torch.randn(1, 81, 64)  # (batch_size, seq_length, input_size)
output_tensor_rnn = rnn_stack(input_tensor_rnn)
print(f"RNNStack output shape: {output_tensor_rnn.shape}")  # Expected: (1, 81, 64)

## Testing Attention
print("\n Testing Attention...")

attention = Attention()
query = torch.randn(1, 10, 64)  # (batch_size, seq_length, hidden_size)
keys = torch.randn(1, 81, 64)  # (batch_size, seq_length, hidden_size)
values = torch.randn(1, 81, 64)  # (batch_size, seq_length, hidden_size)
context = attention(query, keys, values)
print(f"Attention context shape: {context.shape}")  # Expected: (1, 10, 64)

## Testing Tokenizer
print("\n Testing Tokenizer...")

tokenizer = Tokenizer()
input_text = "AH L OW"
elements = tokenizer.split(input_text)
tokens = tokenizer.encode(elements)
text = tokenizer.decode(tokens)
print(f"Tokens: {tokens}") # Expected output : [2, 21, 25]
print(f"Decoded text: {text}")  # Expected output: "AH L OW"

## Testing AcousticEncoder
print("\n Testing AcousticEncoder...")

acoustic_encoder = AcousticEncoder(input_size=64)
input_tensor_encoder = torch.randn(1, 1, 81, 64)
output_tensor_encoder = acoustic_encoder(input_tensor_encoder)
print(f"AcousticEncoder output shape: {output_tensor_encoder.shape}")  # Expected: (1, 81, 64)

## Testing PhoneticEncoder
print("\n Testing PhoneticEncoder...")

phonetic_encoder = PhoneticEncoder(input_size=64)
input_tensor_phonetic = torch.randn(1, 1, 81, 64)
output_tensor_phonetic = phonetic_encoder(input_tensor_phonetic)
print(f"PhoneticEncoder output shape: {output_tensor_phonetic.shape}")

## Testing LinguisticEncoder
print("\n Testing LinguisticEncoder...")

linguistic_encoder = LinguisticEncoder(vocab_size=40, embedding_dim=64, hidden_size=32, output_size=64)
input_tensor_linguistic = torch.randint(0, 40, (1, 10))  # (batch_size, seq_length)
v_output, k_output = linguistic_encoder(input_tensor_linguistic)
print(f"LinguisticEncoder v_output shape: {v_output.shape}")  # Expected: (1, 10, 64)
print(f"LinguisticEncoder k_output shape: {k_output.shape}")  # Expected:

## Testing Decoder
print("\n Testing Decoder...")

decoder = Decoder(input_size=128, output_size=40)
query = torch.randn(1, 10, 64)  # (batch_size, seq_length, input_size)
keys = torch.randn(1, 81, 64)  # (batch_size, seq_length, input_size)
values = torch.randn(1, 81, 64)  # (batch_size, seq_length, input_size)
output = decoder(query, keys, values)
print(f"Decoder output shape: {output.shape}")  # Expected: (1, 10, 40)

## Testing APLModel
print("\n Testing APLModel...")

apl_model = APLModel(input_size=64, vocab_size=40, embedding_dim=64, hidden_size=32, output_size=128, decoder_size=192)
acoustic_input = torch.rand(2, 81, 64)  # (batch_size, seq_length, input_size)
phonetic_input = torch.rand(2, 81, 64)  # (batch_size, seq_length, input_size)
phoneme_input = torch.randint(0, 40, (2, 10))  # (batch_size, seq_length)
print(f"Phoneme input shape: {phoneme_input.shape}")  # Expected: (1, 10)
output_apl = apl_model(acoustic_input, phonetic_input, phoneme_input)
print(f"APLModel output shape : {output_apl.shape}")  # Expected: (2, 10)

print("\n All tests completed successfully.")