---
title: "Getting Started with PyTorch: A Practical Guide"
excerpt: "Learn PyTorch fundamentals through hands-on examples, from basic tensors to building your first neural network."
author: "Alex Rodriguez"
date: "2024-11-11"
tags: ["pytorch", "tutorial", "deep-learning", "beginners"]
image: "/AIML.jpg"
---

# Getting Started with PyTorch: A Practical Guide

PyTorch has become the go-to framework for deep learning research and production. Its dynamic computation graph and intuitive API make it perfect for both beginners and experts. Let's dive into the fundamentals.

## Why PyTorch?

- **Dynamic graphs**: Build models on-the-fly with standard Python control flow
- **Pythonic**: Feels natural if you know Python and NumPy
- **Research-friendly**: Easy to experiment and prototype
- **Production-ready**: TorchScript enables deployment optimization

## Tensors: The Building Blocks

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Creating tensors
x = torch.tensor([1, 2, 3, 4, 5])
y = torch.randn(3, 4)  # Random 3x4 tensor
z = torch.zeros(2, 3, dtype=torch.float32)

# GPU acceleration
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = x.to(device)
```

## Building Your First Neural Network

```python
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize the model
model = SimpleNet(784, 128, 10)
```

## Training Loop

```python
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
```

## Key Concepts

### Autograd
PyTorch's automatic differentiation engine:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # dy/dx = 2x = 4.0
```

### DataLoaders
Efficient data loading and batching:

```python
from torch.utils.data import DataLoader, TensorDataset

# Create dataset
dataset = TensorDataset(features, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Best Practices

1. **Use GPU when available**: Always check `torch.cuda.is_available()`
2. **Set random seeds**: For reproducible results
3. **Monitor gradients**: Watch for vanishing/exploding gradients
4. **Save checkpoints**: Regular model saving during training

This is just the beginning of your PyTorch journey. The framework's flexibility and power make it an excellent choice for any deep learning project.

---

*Coming up: Advanced PyTorch techniques and custom dataset creation*