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
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

# Device configuration
# I do not understand why, but this needs to be at the top of the file
device = (
    "mps" 
    if torch.backends.mps.is_available() 
    else "cuda" 
    if torch.cuda.is_available() 
    else "cpu"
)
print(f"Using device: {device}")

class DiscordDataset(Dataset):
    def __init__(self, gen_folder, altered_folder, transform=None, file_list=None):
        self.transform = transform
        self.data = []
        
        if file_list:
            self.data = [(os.path.join(gen_folder if label == 0 else altered_folder, f), label) for f, label in file_list]
        else:
            print("You've done goofed")
            # # Load generated images (label 0)
            # files = [f for f in os.listdir(gen_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            # for filename in tqdm(files, desc="Loading generated images"):
            #     self.data.append((os.path.join(gen_folder, filename), 0))
                
            # # Load altered images (label 1)
            # files = [f for f in os.listdir(altered_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            # for filename in tqdm(files, desc="Loading altered images"):
            #     self.data.append((os.path.join(altered_folder, filename), 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

class pre_trained_resnet18(nn.Module):
    def __init__(self):
        super(pre_trained_resnet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        return self.resnet(x)

def create_split_datasets(gen_folder, altered_folder, transform_train, transform_val, train_ratio=0.8):
    # Get all file paths
    gen_files = [(f, 0) for f in os.listdir(gen_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    alt_files = [(f, 1) for f in os.listdir(altered_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Split each category separately to maintain balance
    gen_train, gen_val = train_test_split(gen_files, train_size=train_ratio)
    alt_train, alt_val = train_test_split(alt_files, train_size=train_ratio)
    
    # Create datasets with proper splits
    train_dataset = DiscordDataset(gen_folder, altered_folder, 
                                 file_list=gen_train + alt_train,
                                 transform=transform_train)
    val_dataset = DiscordDataset(gen_folder, altered_folder, 
                                file_list=gen_val + alt_val,
                                transform=transform_val)
    
    return train_dataset, val_dataset

def create_data_loaders(gen_folder, altered_folder, batch_size=64, train_split=0.8):
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Simple transforms for validation
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets with respective transforms
    train_dataset, val_dataset = create_split_datasets(gen_folder, altered_folder, transform_train, transform_val, train_ratio=train_split)
    
    # Determine pin_memory based on device
    pin_memory = True if device != "cpu" else False

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, num_epochs=50, patience=5):
    try:
        writer = SummaryWriter(f'runs/training_{time.strftime("%Y%m%d-%H%M%S")}')
        
        # Log model architecture
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        writer.add_graph(model, dummy_input)
        
        best_acc = 0.0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            val_loss_display = 0.0
            val_acc_display = 0.0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # Update progress bar with all metrics
                pbar.set_postfix({
                    'train_loss': f'{loss.item():.4f}',
                    'val_loss': f'{val_loss_display:.4f}',
                    'val_acc': f'{val_acc_display:.4f}'
                })
            
            avg_train_loss = running_loss / len(train_loader)
            
            # Validation phase
            val_loss = 0.0
            correct = 0
            total = 0

            model.eval()
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            val_accuracy = correct / total
            avg_val_loss = val_loss / len(val_loader)
            # Store values for next epoch's display
            val_loss_display = avg_val_loss
            val_acc_display = val_accuracy
            
            # Early stopping check
            if val_accuracy > best_acc:
                best_acc = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after epoch {epoch+1}")
                break
            
            # Log metrics
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Loss/validation', avg_val_loss, epoch)
            writer.add_scalar('Accuracy/validation', val_accuracy, epoch)
        
        return avg_train_loss, avg_val_loss, val_accuracy
    finally:
        writer.close()

def print_training_report(training_time, best_val_acc, best_epoch, history):
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Total training time: {timedelta(seconds=int(training_time))}")
    print(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print("\nTraining History:")
    print(f"Starting training loss: {history['train_loss'][0]:.4f}")
    print(f"Starting validation loss: {history['val_loss'][0]:.4f}")
    print(f"Starting validation accuracy: {history['val_acc'][0]:.4f}")
    print("="*50)

# Usage in main
if __name__ == "__main__":
    # Initialize tracking variables
    start_time = time.time()
    best_val_acc = 0.0
    best_epoch = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Get data loaders and setup model
    train_loader, val_loader = create_data_loaders(
        gen_folder="discord_chats/gen",
        altered_folder="discord_chats/altered",
        batch_size=64
    )
    
    model = pre_trained_resnet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    # Train the model
    avg_train_loss, avg_val_loss, val_accuracy = train_model(model, train_loader, val_loader, num_epochs=10, patience=3)
    
    # Store metrics
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_acc'].append(val_accuracy)
    
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        best_epoch = epoch + 1
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Final Report
    print_training_report(training_time, best_val_acc, best_epoch, history)


# def save_model(model, path='model.pth'):
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'transform': transform_train
#     }, path)
#     print(f"Model saved to {path}")

# def load_model(path='model.pth'):
#     model = pre_trained_resnet18()
#     checkpoint = torch.load(path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
#     return model, checkpoint['transform']

# def predict_image(image_path, model, transform):
#     # Load and preprocess image
#     image = Image.open(image_path).convert('RGB')
#     image = transform(image).unsqueeze(0).to(device)
    
#     # Predict
#     with torch.no_grad():
#         outputs = model(image)
#         _, predicted = torch.max(outputs, 1)
    
#     return "Altered" if predicted.item() == 1 else "Generated"