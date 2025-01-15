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
import optuna # type: ignore # python pytorch_training.py --mode tune

# Hyperparameters
LEARNING_RATE = 0.001 #NOTE: This is slightly to aggressive from my tests on the final run, and it doesn't hit the optimizer as consistently or quickly as it probably should because of this, but these are still the best hyperparameters I have found so far.
FEATURE_REDUCTION = 512 # ResNet50 has 2048 features
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 13
WEIGHT_DECAY = 0.005
NUM_EPOCHS = 40
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

def objective(trial):
    """Optuna objective for hyperparameter optimization"""
    # Suggest hyperparameters
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_int('batch_size', 16, 128, step=16),
        'feature_reduction': trial.suggest_int('feature_reduction', 128, 1024, step=128),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        'lr_factor': trial.suggest_float('lr_factor', 0.1, 0.9, step=0.1),
        'dropout1': trial.suggest_float('dropout1', 0.2, 0.7),
        'dropout2': trial.suggest_float('dropout2', 0.1, 0.5)
    }
    
    try:
        # Create model with trial parameters
        model = pre_trained_resnet50().to(device)
        
        # Get data loaders
        train_loader, val_loader = create_data_loaders(
            gen_folder="discord_chats/gen",
            altered_folder="discord_chats/altered",
            batch_size=params['batch_size']
        )
        
        # Training setup
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), 
                              lr=params['learning_rate'], 
                              weight_decay=params['weight_decay'])
        
        # Train for a few epochs
        history = train_model(
            model, 
            train_loader, 
            val_loader, 
            num_epochs=5,  # Reduced epochs for quick evaluation
            patience=2
        )
        
        # Return best validation accuracy
        return max(history['val_acc'])
        
    except Exception as e:
        print(f"Trial failed: {e}")
        raise optuna.exceptions.TrialPruned()

def log_hyperparameter_study(study):
    """Log hyperparameter optimization results"""
    log_file = "hyperparameter_study.txt"
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Study Date: {timestamp}\n")
        f.write(f"Best Trial Number: {study.best_trial.number}\n")
        f.write(f"Best Parameters:\n")
        for param, value in study.best_params.items():
            f.write(f"{param}: {value}\n")
        f.write(f"{'='*50}\n")

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

# Update model's final layer for single output
class pre_trained_resnet50(nn.Module):
    def __init__(self):
        super(pre_trained_resnet50, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, FEATURE_REDUCTION),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(FEATURE_REDUCTION, 1)  # Changed to 1 output for BCE
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

def calculate_normalization_values(gen_folder, altered_folder):
    """Calculate mean and std values from training dataset"""
    print("Calculating dataset statistics...")
    
    # Get all image paths
    gen_files = [os.path.join(gen_folder, f) for f in os.listdir(gen_folder) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    alt_files = [os.path.join(altered_folder, f) for f in os.listdir(altered_folder) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_files = gen_files + alt_files
    
    # Initialize variables
    mean = torch.zeros(3)
    std = torch.zeros(3)
    
    # First pass: mean
    for img_path in tqdm(all_files, desc="Calculating mean"):
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        mean += img.mean([1, 2])
    mean /= len(all_files)
    
    # Second pass: std
    for img_path in tqdm(all_files, desc="Calculating std"):
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        std += ((img - mean[:, None, None]) ** 2).mean([1, 2])
    std = torch.sqrt(std / len(all_files))
    
    return mean.tolist(), std.tolist()

def create_data_loaders(gen_folder, altered_folder, batch_size=64, train_split=0.8):
    # Calculate normalization values
    mean, std = calculate_normalization_values(gen_folder, altered_folder)
    print(f"Dataset mean: {mean}")
    print(f"Dataset std: {std}")
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Simple transforms for validation
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
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
            labels = labels.float().unsqueeze(1)  # Add dimension to match output
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
                labels = labels.float().unsqueeze(1)  # Add dimension
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = torch.round(torch.sigmoid(outputs))  # Use sigmoid and round for binary classification
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

# Update prediction handling
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
        probabilities = torch.sigmoid(outputs)  # Use sigmoid instead of softmax
        altered_prob = probabilities[0].item()
    
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
            f.write(f"Final Epoch: {final_epoch}\n")
            f.write(f"Final Training Loss: {history['train_loss'][-1]:.4f}\n")
            f.write(f"Final Validation Loss: {history['val_loss'][-1]:.4f}\n")
            f.write(f"Best Validation Accuracy: {max(history['val_acc']):.4f}\n")
            f.write(f"Feature Reduction Size: {FEATURE_REDUCTION}\n")  # Add feature reduction size
            f.write(f"Learning Rate: {optimizer.param_groups[0]['lr']}\n")
            f.write(f"LR Patience: {LR_PATIENCE}\n")
            f.write(f"LR Factor: {LR_FACTOR}\n")
            f.write(f"Batch Size: {BATCH_SIZE}\n")
            f.write(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}\n")
            # Add hyperparameter section
            f.write("\nHyperparameters:\n")
            f.write(f"Learning Rate: {LEARNING_RATE}\n")
            f.write(f"Batch Size: {BATCH_SIZE}\n")
            f.write(f"Feature Reduction: {FEATURE_REDUCTION}\n")
            f.write(f"Weight Decay: {WEIGHT_DECAY}\n")
            f.write(f"LR Factor: {LR_FACTOR}\n")
            f.write(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}\n")
            f.write(f"LR Patience: {LR_PATIENCE}\n")
            
        elif mode == 'predict':
            f.write(f"Model Path: {model_info}\n")
            f.write(f"Image: {args.image}\n")
            f.write(f"Alteration Probability: {prediction_result:.2%}\n")
        
        f.write(f"{'='*50}\n")

# Usage in main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or use a model for altered image detection')
    parser.add_argument('--model', type=str, help='Path to model.pth file')
    parser.add_argument('--mode', choices=['train', 'predict', 'tune'], default='train',
                      help='Mode to run in (default: train)')
    parser.add_argument('--image', type=str, help='Image to predict (required for predict mode)')
    parser.add_argument('--n-trials', type=int, default=10,
                      help='Number of trials for hyperparameter optimization')
    
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
        
    # Update error handling in main
    elif args.mode == 'tune':
        print("\nStarting hyperparameter optimization...")
        study = optuna.create_study(direction="maximize")
        
        try:
            # Create progress bar
            pbar = tqdm(total=args.n_trials, desc="Optimization Progress")
            
            def callback(study, trial):
                pbar.update(1)
                
            # Run optimization with progress tracking
            study.optimize(objective, n_trials=args.n_trials, callbacks=[callback])
            pbar.close()
            
            if study.best_trial:
                print("\nBest trial:")
                trial = study.best_trial
                print(f"Value: {trial.value:.4f}")
                print("\nBest hyperparameters:")
                for key, value in trial.params.items():
                    print(f"{key}: {value}")
                    
                # Log the results
                log_hyperparameter_study(study)
                print("\nResults saved to hyperparameter_study.txt")
            else:
                print("\nNo successful trials completed")
                
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user")
        except Exception as e:
            print(f"\nOptimization failed: {e}")
        
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
        criterion = nn.BCEWithLogitsLoss()
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