import torch
import librosa
from transformers import AutoProcessor, AutoModelForCTC
from torchinfo import summary

# Load the pre-trained model and processor
processor = AutoProcessor.from_pretrained("mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme")
model = AutoModelForCTC.from_pretrained("mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme")

# Load an example audio file
audio_input, sample_rate = librosa.load('data/wav/arctic_a0001.wav', sr=16000)
print("Original sample rate: ", sample_rate)
print("Audio shape: ", audio_input.shape)


input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
print("Input values shape: ", input_values.shape)

# Create fake input values
input_values_fake = torch.randn(1, 16000)

# Print model summary
print("Model summary: \n ")
summary(model, input_size=(1, input_values.shape[1]))
print(model)

# INFERENCE : extract output of feature extraction layers

def capture_feature_encoder_and_transformer_network_outputs(model, input_values):
    # Define a hook to capture the outputs of each convolutional layer
    layer_encoder_outputs = []
    layer_transformer_outputs = []

    def encoder_hook(module, input, output):
        layer_encoder_outputs.append(output)

    def transformer_hook(module, input, output):
        layer_transformer_outputs.append(output)

    # Register the hook to each convolutional layer in the feature encoder
    for layer in model.wav2vec2.feature_extractor.conv_layers:
        layer.register_forward_hook(encoder_hook)

    # Register the hook to each transformer layer
    for layer in model.wav2vec2.encoder.layers:
        layer.register_forward_hook(transformer_hook)

    # Perform a forward pass
    with torch.no_grad():
        outputs = model(input_values)

    # Remove the hooks
    for layer in model.wav2vec2.feature_extractor.conv_layers:
        layer._forward_hooks.clear()
    for layer in model.wav2vec2.encoder.layers:
        layer._forward_hooks.clear()

    return layer_encoder_outputs, layer_transformer_outputs


# Capture the outputs of each layer in the feature encoder
layer_encoder_outputs, layer_transformer_outputs = capture_feature_encoder_and_transformer_network_outputs(model, input_values)

# Print the shape of the outputs for each layer
for i, output in enumerate(layer_encoder_outputs):
    print(f"Layer {i + 1} encoder output shape: {output.shape}")

for i, output in enumerate(layer_transformer_outputs):
    print(f"Layer {i + 1} transformer output shape: {output[0].shape}")  