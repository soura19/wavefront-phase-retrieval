import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import matplotlib.pyplot as plt

# 1. Load the Dataset
print("Loading data...")
data = np.load('coma_dataset.npz')
images = data['images']
labels = data['labels']

# Add a "channel" dimension for PyTorch (NumSamples, Channels, Height, Width)
images = np.expand_dims(images, axis=1)

# Convert to PyTorch Tensors
X = torch.tensor(images, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32).view(-1, 1) # Reshape targets to column

# Split into Training (1600) and Testing (400) sets
dataset = TensorDataset(X, y)
train_set, test_set = torch.utils.data.random_split(dataset, [1600, 400])

# Create batch loaders (Batch size of 32 keeps your 8GB RAM safe)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# 2. Define the Convolutional Neural Network
class ComaCNN(nn.Module):
    def __init__(self):
        super(ComaCNN, self).__init__()
        # First layer looks for basic edges and curves in the diffraction rings
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        # Second layer looks for the asymmetrical "teardrop" shape
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # After pooling twice, our 160x160 image becomes 40x40
        self.fc1 = nn.Linear(16 * 40 * 40, 64)
        # Final output is a single continuous number (the coefficient)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 40 * 40) # Flatten the 2D matrices into a 1D array
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = ComaCNN()

# 3. Setup the Optimizer and Loss Function
criterion = nn.MSELoss() # Mean Squared Error is standard for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Train the Model
epochs = 5
print("Starting training on CPU...")

for epoch in range(epochs):
    start_epoch = time.time()
    model.train()
    running_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()      # Clear old gradients
        outputs = model(batch_X)   # Forward pass (guess the coefficient)
        loss = criterion(outputs, batch_y) # Calculate how wrong it was
        loss.backward()            # Backpropagation (calculate adjustments)
        optimizer.step()           # Update the network weights
        
        running_loss += loss.item()
        
    avg_train_loss = running_loss / len(train_loader)
    
    # 5. Evaluate on Test Set
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            test_loss += loss.item()
            
    avg_test_loss = test_loss / len(test_loader)
    time_taken = time.time() - start_epoch
    
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Time: {time_taken:.1f}s")

print("Training complete! Your AI can now detect coma aberrations.")

# 6. Save the Model Weights
torch.save(model.state_dict(), 'coma_cnn.pth')
print("Model saved to 'coma_cnn.pth'")

# 7. Visualize True vs Predicted (The Physics Proof)
model.eval()
true_vals = []
pred_vals = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        preds = model(batch_X)
        true_vals.extend(batch_y.numpy().flatten())
        pred_vals.extend(preds.numpy().flatten())

# Plot the results
plt.figure(figsize=(6, 6))
plt.scatter(true_vals, pred_vals, alpha=0.5, color='teal', label='AI Predictions')
plt.plot([-4, 4], [-4, 4], color='red', linestyle='--', label='Perfect Accuracy')
plt.title("AI Performance: True vs Predicted Coma Aberration")
plt.xlabel("True Zernike Coefficient")
plt.ylabel("AI Predicted Coefficient")
plt.legend()
plt.grid(True)
plt.show()