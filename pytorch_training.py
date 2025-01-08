import os
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
from PIL import Image

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

# Training loop
for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images, labels = images, labels
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Evaluate
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Validation Accuracy:", correct / total)