from PIL import Image, ImageDraw, ImageFont
import random
import os

# Paths to required assets
FONT_PATH_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"  # Update for your system
FONT_PATH_REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"    # Update for your system
MESSAGES_FILE = "message_examples.txt"  # Path to the file containing message examples

# Path to the folder containing avatar images
avatars_folder = "images/avatars"

# TO REMOVE. This is currently not used in the improved generator, but we may use it later.
AVATAR_PATH = "images/placeholder_avatar.png"  # Path to a default avatar image

# Chat appearance settings
BACKGROUND_COLOR = (54, 57, 63)  # Discord background color
TEXT_COLOR = (220, 221, 222)     # Discord message text color
USERNAME_COLOR = (88, 101, 242)  # Discord username color
TIMESTAMP_COLOR = (163, 166, 170)  # Discord timestamp color
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

# Generate a synthetic Discord chat
def generate_discord_chat(messages, avatar_path, output_path):
    # Calculate image height dynamically based on the number of messages
    total_height = (LINE_HEIGHT + MESSAGE_PADDING) * len(messages) + MESSAGE_PADDING
    img = Image.new("RGB", (IMAGE_WIDTH, total_height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    y_offset = MESSAGE_PADDING  # Initial padding from top

    # Create and paste messages
    for username, discriminator, message, timestamp in messages:
        # Create circular avatar
        avatar = create_circular_avatar(avatar_path)

        # Paste avatar
        img.paste(avatar, (20, y_offset), avatar)

        # Draw username, discriminator, and timestamp
        username_text = f"{username}#{discriminator}"
        username_width = draw.textbbox((0, 0), username_text, font=font_bold)[2]
        timestamp_width = draw.textbbox((0, 0), timestamp, font=font_regular)[2]
        timestamp_x = username_width + 100

        # Ensure username and timestamp fit in the same line
        max_username_width = 300
        if username_width > max_username_width:
            username_text = username_text[:max_username_width] + "..."

        draw.text((80, y_offset), username_text, fill=USERNAME_COLOR, font=font_bold)
        draw.text((timestamp_x, y_offset), timestamp, fill=TIMESTAMP_COLOR, font=font_regular)

        # Draw message text
        draw.text((80, y_offset + LINE_HEIGHT // 2), message, fill=TEXT_COLOR, font=font_regular)

        # Update y_offset for next message
        y_offset += LINE_HEIGHT + MESSAGE_PADDING

    # Save the image
    img.save(output_path)
    print(f"Generated Discord chat saved to {output_path}")

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
def generate_multiple_chats(output_dir, num_chats, num_messages_per_chat, message_examples):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_chats):
        messages = generate_random_messages(num_messages_per_chat, message_examples)
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
    generate_multiple_chats(output_dir="discord_chats/gen/", num_chats=1000, num_messages_per_chat=10, message_examples=message_examples)