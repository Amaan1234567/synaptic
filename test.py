import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

# Load the digits dataset
digits = load_digits()

# Preprocess data
X = digits.data.astype(float) / 255  # Normalize pixel values
y = digits.target

# Convert data to tensors
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()


# Custom dataset class for iterating over single samples
class DigitDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# Create the dataset instance
dataset = DigitDataset(X_tensor, y_tensor)

# Create dataloaders with batch size 1 for training and testing (no shuffling for testing)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False)  # Don't shuffle for testing


# Define the neural network model
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,30),
            nn.ReLU(),
            nn.Linear(30, 10),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


# Create the model instance
model = DigitClassifier()

# Define loss function and optimizer (SGD with learning rate 0.01)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
num_epochs = 10

for epoch in range(num_epochs):
    for i, (data, target) in enumerate(train_loader):
        # Forward pass
        y_pred = model(data)
        loss = criterion(y_pred, target)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print training loss after each epoch
    print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Test the model
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        y_pred = model(data)
        predicted = torch.argmax(y_pred, dim=1)  # Get predicted class
        correct += (predicted == target).item()
        total += 1

# Calculate and print accuracy
accuracy = correct / total
print(f'Test Accuracy: {accuracy:.4f}')

# Alternative way to calculate accuracy using sklearn.metrics
y_true = []
y_pred = []
with torch.no_grad():
    for data, target in test_loader:
        y_pred.append(torch.argmax(model(data), dim=1).item())
        y_true.append(target.item())

test_accuracy = accuracy_score(y_true, y_pred)
print(f'Test Accuracy (using sklearn): {test_accuracy:.4f}')
