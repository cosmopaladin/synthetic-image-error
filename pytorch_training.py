import os
import torch
from tqdm import tqdm # type: ignore
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import time
from datetime import timedelta

class DiscordDataset(Dataset):
    def __init__(self, gen_folder, altered_folder, transform=None):
        self.transform = transform
        self.data = []
        
        # Load generated images (label 0)
        files = [f for f in os.listdir(gen_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for filename in tqdm(files, desc="Loading generated images"):
            self.data.append((os.path.join(gen_folder, filename), 0))
            
        # Load altered images (label 1)
        files = [f for f in os.listdir(altered_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for filename in tqdm(files, desc="Loading altered images"):
            self.data.append((os.path.join(altered_folder, filename), 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Create dataset from discord_chats folders
dataset = DiscordDataset("discord_chats/gen", "discord_chats/altered", transform=transform)

# Split into train/validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize metric tracking
start_time = time.time()
history = {
    'train_loss': [],
    'val_accuracy': []
}

# Training loop with progress bar and metrics
for epoch in range(10):
    model.train()
    running_loss = 0.0
    with tqdm(train_loader, desc=f'Epoch {epoch+1}/10') as pbar:
        for images, labels in pbar:
            images, labels = images, labels
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{running_loss/len(train_loader):.4f}'})
    
    # Store training loss
    epoch_loss = running_loss/len(train_loader)
    history['train_loss'].append(epoch_loss)
    
    # Validation after each epoch
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_acc = correct / total
    history['val_accuracy'].append(val_acc)
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}")

# Final Report
training_time = time.time() - start_time
print("\n=== Training Complete ===")
print(f"Total training time: {timedelta(seconds=int(training_time))}")
print("\nFinal Metrics:")
print(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")
print(f"Final training loss: {history['train_loss'][-1]:.4f}")
print(f"\nTraining loss by epoch: {[f'{loss:.4f}' for loss in history['train_loss']]}")
print(f"Validation accuracy by epoch: {[f'{acc:.4f}' for acc in history['val_accuracy']]}")

def save_model(model, path='model.pth'):
    torch.save({
        'model_state_dict': model.state_dict(),
        'transform': transform
    }, path)
    print(f"Model saved to {path}")

def load_model(path='model.pth'):
    model = SimpleCNN()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['transform']

def predict_image(image_path, model, transform):
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return "Altered" if predicted.item() == 1 else "Generated"

# Add after training loop
save_model(model)

# # Example usage for prediction
# if __name__ == "__main__":
#     # Training code here
#     ...existing code...
    
#     # Save model after training
#     save_model(model)
    
#     # Example of loading and using model
#     loaded_model, transform = load_model()
#     result = predict_image("path_to_test_image.jpg", loaded_model, transform)
#     print(f"Prediction: {result}")