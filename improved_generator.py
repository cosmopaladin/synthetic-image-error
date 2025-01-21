from PIL import Image, ImageDraw, ImageFont
import random
import os

# Paths to required assets
FONT_PATH_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"  # Update for your system
FONT_PATH_REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"    # Update for your system
MESSAGES_FILE = "message_examples.txt"  # Path to the file containing message examples

# Path to the folder containing avatar images
avatars_folder = "images/avatars"

# CLEAN THIS SHIT UP
# TO REMOVE. This is used in the code, but not in the final images.
# There is some weird spaghetti code going on here
AVATAR_PATH = "images/placeholder_avatar.png"  # Path to a default avatar image

FONT_SIZE = 20
LINE_HEIGHT = 50  # Line height for messages
AVATAR_SIZE = 40  # Size of circular avatar
IMAGE_WIDTH = 650  # 1.3 aspect ratio
MESSAGE_PADDING = 20  # Padding between messages

# Load fonts
font_bold = ImageFont.truetype(FONT_PATH_BOLD, FONT_SIZE)
font_regular = ImageFont.truetype(FONT_PATH_REGULAR, FONT_SIZE - 2)

# Helper function to create circular avatars
def create_circular_avatar(avatar_path, size=AVATAR_SIZE):
  # Get a list of all files in the folder
  avatar_files = [f for f in os.listdir(avatars_folder) if os.path.isfile(os.path.join(avatars_folder, f))]
  
  # Ensure the folder isn't empty
  if not avatar_files:
      raise FileNotFoundError(f"No files found in {avatars_folder}.")
  # Choose a random file
  random_avatar = random.choice(avatar_files)

  # Full path to the selected file
  random_avatar_path = os.path.join(avatars_folder, random_avatar)
  
  avatar = Image.open(random_avatar_path).convert("RGBA")
  avatar = avatar.resize((size, size), Image.Resampling.LANCZOS)

  # Create circular mask
  mask = Image.new("L", (size, size), 0)
  draw = ImageDraw.Draw(mask)
  draw.ellipse((0, 0, size-4, size-4), fill=255)

  # Apply mask to avatar
  circular_avatar = Image.new("RGBA", (size, size))
  circular_avatar.paste(avatar, (0, 0), mask=mask)
  return circular_avatar

def calculate_contrast_ratio(color1, color2):
    """Calculate contrast ratio between two colors"""
    def luminance(rgb):
        rgb = [x/255 for x in rgb]
        rgb = [x/12.92 if x <= 0.03928 else ((x+0.055)/1.055)**2.4 for x in rgb]
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    
    l1, l2 = luminance(color1), luminance(color2)
    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)

def generate_chat_colors(usernames):
    """Generate color scheme with unique username colors for light/dark theme"""
    # Theme configurations
    themes = {
        'dark': {
            'background_range': (20, 40),
            'text_range': (200, 255),
            'timestamp_range': (140, 160),
            'username_range': (100, 255),
            'min_contrast': 4.5
        },
        'light': {
            'background_range': (240, 255),
            'text_range': (20, 60),
            'timestamp_range': (100, 120),
            'username_range': (0, 150),
            'min_contrast': 4.0
        }
    }
    
    # Select random theme
    theme_type = random.choice(list(themes.keys()))
    theme = themes[theme_type]
    
    # Generate base colors
    background = (random.randint(*theme['background_range']),) * 3
    
    # Generate text color with contrast check
    while True:
        text = (random.randint(*theme['text_range']),) * 3
        if calculate_contrast_ratio(background, text) >= theme['min_contrast']:
            break
    
    # Generate timestamp color with contrast check
    while True:
        timestamp = (random.randint(*theme['timestamp_range']),) * 3
        if calculate_contrast_ratio(background, timestamp) >= theme['min_contrast']:
            break
    
    # Generate unique username colors with contrast check
    username_colors = {}
    for username in usernames:
        while True:
            color = tuple(
                random.randint(*theme['username_range'])
                for _ in range(3)
            )
            if calculate_contrast_ratio(background, color) >= theme['min_contrast']:
                username_colors[username] = color
                break
    
    return {
        'background': background,
        'text': text,
        'timestamp': timestamp,
        'usernames': username_colors,
        'theme': theme_type
    }

# Generate a synthetic Discord chat
def generate_discord_chat(messages, avatar_path, output_path):
    # Get unique usernames from messages
    usernames = {username for username, _, _, _ in messages}
    
    # Get random colors with unique username colors
    colors = generate_chat_colors(usernames)
    
    # Create image
    total_height = (LINE_HEIGHT + MESSAGE_PADDING) * len(messages) + MESSAGE_PADDING
    img = Image.new("RGB", (IMAGE_WIDTH, total_height), colors['background'])
    draw = ImageDraw.Draw(img)
    y_offset = MESSAGE_PADDING

    for username, discriminator, message, timestamp in messages:
        avatar = create_circular_avatar(avatar_path)
        img.paste(avatar, (20, y_offset), avatar)
        
        username_text = f"{username}#{discriminator}"
        username_width = draw.textbbox((0, 0), username_text, font=font_bold)[2]
        timestamp_x = username_width + 100

        # Use unique color for each username
        draw.text((80, y_offset), username_text, 
                 fill=colors['usernames'][username], font=font_bold)
        draw.text((timestamp_x, y_offset), timestamp, 
                 fill=colors['timestamp'], font=font_regular)
        draw.text((80, y_offset + LINE_HEIGHT // 2), message, 
                 fill=colors['text'], font=font_regular)

        y_offset += LINE_HEIGHT + MESSAGE_PADDING

    img.save(output_path)
    print(f"Saved {colors['theme']} theme chat to: {output_path}")

# Load messages from a file
def load_messages_from_file(file_path):
    if not os.path.exists(file_path):
        print(f"Messages file not found at {file_path}. Please provide a valid file.")
        exit()

    with open(file_path, "r") as f:
        messages = [line.strip() for line in f if line.strip()]
    return messages

# Generate random messages
def generate_random_messages(num_messages, message_examples):
    messages = []
    for _ in range(num_messages):
        username = f"User{random.randint(1, 50)}"
        discriminator = random.randint(1000, 9999)
        message = random.choice(message_examples)
        timestamp = f"{random.randint(1, 12)}:{random.randint(0, 59):02d} {'AM' if random.randint(0, 1) == 0 else 'PM'}"
        messages.append((username, discriminator, message, timestamp))
    return messages

# Generate multiple chats
def generate_multiple_chats(output_dir, num_chats, message_examples):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_chats):
        # Randomly choose number of messages between 2 and 10
        num_messages = random.randint(2, 10)
        messages = generate_random_messages(num_messages, message_examples)
        output_path = os.path.join(output_dir, f"discord_chat_{i + 1}.png")
        generate_discord_chat(messages, avatar_path=AVATAR_PATH, output_path=output_path)

# Example usage
if __name__ == "__main__":
    # Ensure you have an avatar image at the specified path
    if not os.path.exists(AVATAR_PATH):
        print(f"Avatar not found at {AVATAR_PATH}. Please provide a valid avatar path.")
        exit()

    # Load message examples from a file
    message_examples = load_messages_from_file(MESSAGES_FILE)

    # Generate 5 synthetic chats, each with 10 messages
    generate_multiple_chats(output_dir="discord_chats/gen/", num_chats=10000, message_examples=message_examples)