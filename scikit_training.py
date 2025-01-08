import os
import cv2 # type: ignore
import numpy as np
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import accuracy_score # type: ignore

def load_images(gen_folder, altered_folder):
    images, labels = [], []
    
    # Load unaltered/generated images (label 0)
    for filename in os.listdir(gen_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(gen_folder, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                images.append(img.flatten())
                labels.append(0)
    
    # Load altered images (label 1)
    for filename in os.listdir(altered_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(altered_folder, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                images.append(img.flatten())
                labels.append(1)
    
    return np.array(images), np.array(labels)

# Load data from discord_chats folders
X, y = load_images("discord_chats/gen", "discord_chats/altered")

# Split into train/validation sets
train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(train_x, train_y)

# Evaluate
val_pred = clf.predict(val_x)
print("Validation Accuracy:", accuracy_score(val_y, val_pred))