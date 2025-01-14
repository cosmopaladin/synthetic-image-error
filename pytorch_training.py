import os
import torch # type: ignore
from tqdm import tqdm # type: ignore
import torchvision.transforms as transforms # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from torchvision.models import resnet50, ResNet50_Weights # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from PIL import Image # type: ignore
import time
from datetime import timedelta
import torch.nn.init as init # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import argparse

# Hyperparameters
LEARNING_RATE = 0.005
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 10
WEIGHT_DECAY = 0.005
NUM_EPOCHS = 30
LR_PATIENCE = 3
LR_FACTOR = 0.5 #LEARNING_RATE * LR_FACTOR = new learning rate

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

class pre_trained_resnet50(nn.Module):
    def __init__(self):
        super(pre_trained_resnet50, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),  # ResNet50 has 2048 features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
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
    best_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            pbar.set_postfix({
                'train_loss': f'{loss.item():.4f}'
            })
        
        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
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
        
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        
        print(f'\nEpoch: {epoch+1}')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.4f}')
        
        # Early stopping check
        # NOTE this will not save every checkpoint, but only ones which are improving or the inal checkpoint before an early stop
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            patience_counter = 0
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            save_checkpoint(
                model,
                epoch,
                val_accuracy,
                optimizer,
                f'model_checkpoint_acc{val_accuracy:.4f}_{timestamp}.pth'
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                print(f'Best validation accuracy: {best_acc:.4f}')
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                save_checkpoint(
                    model,
                    epoch,
                    val_accuracy,
                    optimizer,
                    f'model_checkpoint_acc{val_accuracy:.4f}_{timestamp}.pth'
                )
                break
    
    return history

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

def save_checkpoint(model, epoch, val_accuracy, optimizer, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_accuracy,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(checkpoint_path):
    """Load a saved model checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = pre_trained_resnet50()  # Use ResNet50
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    return model, checkpoint

def predict_image(model, image_path):
    """Predict if an image has been altered"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        altered_prob = probabilities[0][1].item()
    
    return altered_prob

def log_run_info(mode, model_info, runtime=None, history=None, prediction_result=None, final_epoch=None):
    """Log run information to a report file"""
    log_file = "training_report.txt"
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Run Date: {timestamp}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Device: {device}\n")
        
        if mode == 'train':
            f.write(f"Training Duration: {timedelta(seconds=int(runtime))}\n")
            f.write(f"Final Epoch: {final_epoch}\n")  # Add final epoch
            f.write(f"Final Training Loss: {history['train_loss'][-1]:.4f}\n")
            f.write(f"Final Validation Loss: {history['val_loss'][-1]:.4f}\n")
            f.write(f"Best Validation Accuracy: {max(history['val_acc']):.4f}\n")
            f.write(f"Learning Rate: {optimizer.param_groups[0]['lr']}\n")
            f.write(f"LR Patience: {LR_PATIENCE}\n")
            f.write(f"LR Factor: {LR_FACTOR}\n")
            f.write(f"Batch Size: {BATCH_SIZE}\n")
            f.write(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}\n")
            
        elif mode == 'predict':
            f.write(f"Model Path: {model_info}\n")
            f.write(f"Image: {args.image}\n")
            f.write(f"Alteration Probability: {prediction_result:.2%}\n")
        
        f.write(f"{'='*50}\n")

# Usage in main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or use a model for altered image detection')
    parser.add_argument('--model', type=str, help='Path to model.pth file')
    parser.add_argument('--mode', choices=['train', 'predict'], default='train',
                      help='Mode to run in (default: train)')
    parser.add_argument('--image', type=str, help='Image to predict (required for predict mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'predict':
        if not args.model:
            parser.error("--model is required for predict mode")
        if not args.image:
            parser.error("--image is required for predict mode")
            
        model, checkpoint = load_checkpoint(args.model)
        prob = predict_image(model, args.image)
        print(f"\nPrediction for {args.image}:")
        print(f"Probability of being altered: {prob:.2%}")
        log_run_info('predict', args.model, prediction_result=prob)
        
    else:  # train mode
        # Initialize tracking variables
        start_time = time.time()
        best_val_acc = 0.0
        best_epoch = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Setup model and training
        if args.model:  # Continue training existing model
            model, checkpoint = load_checkpoint(args.model)
            print(f"Continuing training from checkpoint: {args.model}")
        else:  # Train new model
            model = pre_trained_resnet50().to(device)  # Use ResNet50
            
        # Get data loaders
        train_loader, val_loader = create_data_loaders(
            gen_folder="discord_chats/gen",
            altered_folder="discord_chats/altered",
            batch_size=BATCH_SIZE
        )
        
        # Training parameters
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), 
                              lr=LEARNING_RATE, 
                              weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=LR_FACTOR, 
            patience=LR_PATIENCE
        )
        
        # Train the model
        history = train_model(
            model, 
            train_loader, 
            val_loader, 
            num_epochs=NUM_EPOCHS, 
            patience=EARLY_STOPPING_PATIENCE
        )
        
        # Calculate training time and print report
        training_time = time.time() - start_time
        final_epoch = len(history['train_loss'])  # Get final epoch number
        print_training_report(training_time, best_val_acc, best_epoch, history)
        
        # Log run information with final epoch
        log_run_info('train', None, training_time, history, final_epoch=final_epoch)