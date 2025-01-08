import os
import cv2 # type: ignore
import numpy as np
from tqdm import tqdm # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import accuracy_score # type: ignore

def load_images(gen_folder, altered_folder):
    images, labels = [], []
    
    # Load unaltered/generated images (label 0)
    gen_files = [f for f in os.listdir(gen_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for filename in tqdm(gen_files, desc="Loading generated images"):
        filepath = os.path.join(gen_folder, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128))
            images.append(img.flatten())
            labels.append(0)
    
    # Load altered images (label 1)
    altered_files = [f for f in os.listdir(altered_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for filename in tqdm(altered_files, desc="Loading altered images"):
        filepath = os.path.join(altered_folder, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128))
            images.append(img.flatten())
            labels.append(1)
    
    return np.array(images), np.array(labels)

# Load data from discord_chats folders
X, y = load_images("discord_chats/gen", "discord_chats/altered")

# Split into train/validation sets with stratification
train_x, val_x, train_y, val_y = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # This ensures balanced classes in both splits
)

# Print distribution of classes
print("Training set distribution:")
print(f"Generated images: {sum(train_y == 0)}")
print(f"Altered images: {sum(train_y == 1)}")
print("\nValidation set distribution:")
print(f"Generated images: {sum(val_y == 0)}")
print(f"Altered images: {sum(val_y == 1)}")

# Train classifier with progress bar
clf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=0)
with tqdm(total=100, desc="Training Random Forest") as pbar:
    clf.fit(train_x, train_y)
    pbar.update(100)

# Evaluate
val_pred = clf.predict(val_x)
print("\nValidation Accuracy:", accuracy_score(val_y, val_pred))