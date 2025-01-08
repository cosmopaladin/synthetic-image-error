import os
import random
import shutil

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
    
    # Move selected images
    for filename in to_move:
        src = os.path.join(gen_path, filename)
        dst = os.path.join(to_alter_path, filename)
        shutil.move(src, dst)
        
    print(f"Moved {len(to_move)} images to {to_alter_path}")

if __name__ == "__main__":
    distribute_images()