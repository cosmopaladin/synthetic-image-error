import os
from PIL import Image
import pytesseract # type: ignore
import numpy as np
import random

# Function to add reduced and tighter noise to the borders of a detected text region
def add_noise_to_borders(image_path):
    # Open the image
    image = Image.open(image_path)
    image_np = np.array(image)
    height, width, _ = image_np.shape  # Get image dimensions
    
    # Use pytesseract to detect the text regions (bounding boxes)
    d = pytesseract.image_to_boxes(image)
    
    # Iterate through each character's bounding box
    for box in d.splitlines():
        b = box.split()
        x1, y1, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        
        # Ensure coordinates are within image bounds
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        
        # Tighten the noise area and reduce noise probability (90% reduction)
        noise_margin = 2  # Reduced margin around text
        noise_probability = 0.02  # Reduced noise probability
        
        # Add noise around the borders of each text box
        for i in range(x1 - noise_margin, x2 + noise_margin):  # Reduced margin for tighter noise
            for j in range(y1 - noise_margin, y2 + noise_margin):  # Reduced margin for tighter noise
                # Ensure indices are within valid range
                if 0 <= i < width and 0 <= j < height:
                    if random.random() < noise_probability:  # Reduced probability of noise
                        image_np[j, i] = [random.randint(0, 255) for _ in range(3)]
    
    # Convert back to PIL image
    noisy_image = Image.fromarray(image_np)
    
    return noisy_image

# Function to process all images in a folder and save the altered images
def process_images(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    for filename in os.listdir(input_folder):
        # Process only image files (you can expand the list of image extensions if needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            
            # Add noise to the image
            noisy_image = add_noise_to_borders(image_path)
            
            # Save the altered image to the output folder
            output_path = os.path.join(output_folder, f"altered_{filename}")
            noisy_image.save(output_path)
            print(f"Processed and saved: {output_path}")

# Example usage
input_folder = "discord_chats/to_alter"
output_folder = "discord_chats/altered"

# Process all images in the input folder and save the altered versions
process_images(input_folder, output_folder)