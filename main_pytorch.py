import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import nn, optim
import numpy as np
import os

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transformation, similar to ImageDataGenerator in TensorFlow
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Assuming RGB channels, adjust if needed
])

# Loading datasets (replacing flow_from_directory with ImageFolder and DataLoader in PyTorch)
# Loading datasets (apply same transformation to validation data as well)
train_data = datasets.ImageFolder('C:/Users/sharr/OneDrive/Desktop/NLP pytorch/train', transform=data_transform)
val_data = datasets.ImageFolder('C:/Users/sharr/OneDrive/Desktop/NLP pytorch/val', transform=data_transform)  # Fix here

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)


# Define the model architecture in PyTorch
class FoodClassificationModel(nn.Module):
    def __init__(self):
        super(FoodClassificationModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 5)  # Assuming 5 output classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model, move it to GPU if available, define optimizer and loss function
model = FoodClassificationModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop (replacing TensorFlow's model.fit)
num_epochs = 10  # Example number of epochs
best_val_accuracy = 0.0  # To keep track of the best validation accuracy

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU if available

        optimizer.zero_grad()  # Zero out gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

    # Validation loop (replacing TensorFlow's model.evaluate)
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU if available
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Compute validation loss
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f"Validation Accuracy: {val_accuracy}%, Validation Loss: {val_loss / len(val_loader)}")

    # Save the model checkpoint if validation accuracy improves
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_food_model.pth')
        print(f"Model saved at epoch {epoch+1} with validation accuracy {val_accuracy}%")

# Save the final model at the end of training
torch.save(model.state_dict(), 'final_food_model.pth')
