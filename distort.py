import os
from PIL import Image, ImageFilter, ImageEnhance
import pytesseract # type: ignore
import numpy as np
import random
from io import BytesIO
import cv2  # type: ignore

def subtle_distort(image_np, box, intensity=0.3):
    x1, y1, x2, y2 = box
    # Extend box slightly
    margin = 3
    x1, x2 = max(0, x1-margin), min(image_np.shape[1], x2+margin)
    y1, y2 = max(0, y1-margin), min(image_np.shape[0], y2+margin)
    
    region = image_np[y1:y2, x1:x2].copy()
    original_region = region.copy()
    
    # Available distortion types
    distortions = ['blur', 'noise', 'compression', 'color_shift']
    
    # Randomly select 1-4 unique distortions
    num_distortions = random.randint(1, 4)
    selected_distortions = random.sample(distortions, num_distortions)
    
    # Apply each selected distortion
    for distortion in selected_distortions:
        if distortion == 'blur':
            region_img = Image.fromarray(region)
            region = np.array(region_img.filter(ImageFilter.GaussianBlur(radius=0.5)))
        
        elif distortion == 'noise':
            noise = np.random.normal(0, 2, region.shape)
            region = np.clip(region + noise * intensity, 0, 255).astype(np.uint8)
        
        elif distortion == 'compression':
            img = Image.fromarray(region)
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)
            region = np.array(Image.open(buffer))
        
        elif distortion == 'color_shift':
            shift = np.random.uniform(-5, 5, 3) * intensity
            region = np.clip(region + shift, 0, 255).astype(np.uint8)
    
    # Final blend with original
    alpha = np.random.uniform(0.7, 0.9)
    image_np[y1:y2, x1:x2] = cv2.addWeighted(
        image_np[y1:y2, x1:x2], 1-alpha,
        region, alpha, 0
    )
    
    return image_np

# Function to add reduced and tighter noise to the borders of a detected text region
def add_noise_to_borders(image_path):
    # Open the image
    image = Image.open(image_path)
    image_np = np.array(image)
    
    # Get text boxes
    d = pytesseract.image_to_boxes(image)
    
    # Process each text region
    for box in d.splitlines():
        b = box.split()
        box_coords = [int(b[1]), int(b[2]), int(b[3]), int(b[4])]
        
        # Apply subtle distortion
        image_np = subtle_distort(image_np, box_coords)
    
    return Image.fromarray(image_np)

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