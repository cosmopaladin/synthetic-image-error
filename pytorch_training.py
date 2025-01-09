import os
import torch
from tqdm import tqdm # type: ignore
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import time
from datetime import timedelta
import torch.nn.init as init

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

class ImprovedModel(nn.Module):
    def __init__(self):
        super(ImprovedModel, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
    #     self._initialize_weights()
    
    # def _initialize_weights(self):
    #     for m in self.resnet.fc.modules():
    #         if isinstance(m, nn.Linear):
    #             init.kaiming_normal_(m.weight)
    #             init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.resnet(x)



def save_model(model, path='model.pth'):
    torch.save({
        'model_state_dict': model.state_dict(),
        'transform': transform_train
    }, path)
    print(f"Model saved to {path}")

def load_model(path='model.pth'):
    model = ImprovedModel()
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

# Train model
if __name__ == "__main__":
    # Training parameters
    num_epochs = 50
    batch_s = 64

    # Early stopping
    best_acc = 0.0
    patience = 5
    patience_counter = 0

    # Enhanced transforms with more aggressive augmentation
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),  # Larger initial size for random crops
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.15, 0.15),
            scale=(0.8, 1.2),
            shear=10
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Create dataset from discord_chats folders
    dataset = DiscordDataset("discord_chats/gen", "discord_chats/altered", transform=transform_train)

    # Split into train/validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_s, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_s)

    # Device setup
    device = (
        "mps" 
        if torch.backends.mps.is_available()
        else "cuda" 
        if torch.cuda.is_available() 
        else "cpu"
    )
    print(f"Using device: {device}")

    # Model setup
    model = ImprovedModel()
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001) # learning rate. If too small it can get stuck in a local min. If too big it will never reach the minimum. Hyper parameter
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)

    # Gradient clipping
    max_grad_norm = 1.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # Initialize metric tracking
    start_time = time.time()
    history = {
        'train_loss': [],
        'val_accuracy': []
    }

    # Training loop with progress bar and metrics
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
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
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = correct / total
        history['val_accuracy'].append(acc)
        scheduler.step(acc)
        
        # Early stopping check
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Val Acc: {acc:.4f}")

    # Final Report
    training_time = time.time() - start_time
    print("\n=== Training Complete ===")
    print(f"Total training time: {timedelta(seconds=int(training_time))}")
    print("\nFinal Metrics:")
    print(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"\nTraining loss by epoch: {[f'{loss:.4f}' for loss in history['train_loss']]}")
    print(f"Validation accuracy by epoch: {[f'{acc:.4f}' for acc in history['val_accuracy']]}")
    # Save model after training
    save_model(model)
    
    # Example of loading and using model
    loaded_model, transform = load_model()
    result = predict_image("path_to_test_image.jpg", loaded_model, transform)
    print(f"Prediction: {result}")