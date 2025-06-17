import torch
from sub import CNNStack, RNNStack

test_input = torch.randn(1,81,128)  # Example input tensor
model = RNNStack(input_size=128, hidden_size=128)  # Initialize the model with appropriate sizes

output = model(test_input)
print("Output shape:", output.shape)  # Print the shape of the output tensor