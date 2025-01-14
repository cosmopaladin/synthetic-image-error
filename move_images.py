import os
import random
import shutil
from tqdm import tqdm # type: ignore

def clean_directories():
    # Paths to clean
    paths = ["discord_chats/to_alter", "discord_chats/altered"]
    
    for path in paths:
        if os.path.exists(path):
            files = os.listdir(path)
            for file in files:
                file_path = os.path.join(path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"Cleaned {path}")

def distribute_images():
    # Setup paths
    gen_path = "discord_chats/gen"
    to_alter_path = "discord_chats/to_alter"
    
    # Create to_alter directory if it doesn't exist
    os.makedirs(to_alter_path, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(gen_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Calculate number of images to move
    num_to_move = len(image_files) // 2
    
    # Randomly select images to move
    to_move = random.sample(image_files, num_to_move)
    
    # Move and delete selected images
    for filename in tqdm(to_move, desc="Processing files"):
        src = os.path.join(gen_path, filename)
        dst = os.path.join(to_alter_path, filename)
        try:
            shutil.copy2(src, dst)  # Copy file first
            os.remove(src)  # Then remove original
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
        
    print(f"Processed {len(to_move)} images: copied to {to_alter_path} and removed from {gen_path}")

if __name__ == "__main__":
    print("Cleaning directories...")
    clean_directories()
    print("\nStarting image distribution...")
    distribute_images()