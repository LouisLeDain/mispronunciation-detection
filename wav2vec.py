import torch
from transformers import Wav2Vec2Phoneme, Wav2Vec2Processor

# Load pre-trained model and processor
model_name = "facebook/wav2vec2-phoneme-base"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2PhonemeModel.from_pretrained(model_name)

# Load and preprocess audio file
audio_file = "data/wav/arctic_a0001.wav"
input_values = processor(audio_file, return_tensors="pt", sampling_rate=16000).input_values

# Forward pass to retrieve layer outputs
with torch.no_grad():
    outputs = model(input_values, output_hidden_states=True)

# Extract hidden states for each layer
hidden_states = outputs.hidden_states

# Print the shape of hidden states for each layer
for i, layer_hidden_states in enumerate(hidden_states):
    print(f"Layer {i}: {layer_hidden_states.shape}")
