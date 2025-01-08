import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
def load_images(folder):
    images, labels = [], []
    for label, subfolder in enumerate(["unaltered", "altered"]):
        path = os.path.join(folder, subfolder)
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
            img = cv2.resize(img, (128, 128))  # Resize for uniformity
            images.append(img.flatten())  # Flatten to a vector
            labels.append(label)
    return np.array(images), np.array(labels)

train_x, train_y = load_images("synthetic_data/train")
val_x, val_y = load_images("synthetic_data/val")

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(train_x, train_y)

# Evaluate
val_pred = clf.predict(val_x)
print("Validation Accuracy:", accuracy_score(val_y, val_pred))